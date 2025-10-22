"""
Code Agent Context Management System

Intelligent context window management with pruning, summarization, and relevance scoring
to prevent token overflow while maintaining conversation quality.

Features:
- **Token Counting**: Accurate token counting using tiktoken
- **Pruning Strategies**: Recency, Importance, Sliding Window, Relevance-based
- **Relevance Scoring**: Keyword-based semantic similarity
- **Summarization**: Incremental conversation summarization
- **Context Orchestration**: Automatic pruning/summarization triggers
- **Checkpointing**: Save/restore context state
- **Health Metrics**: Context statistics and monitoring

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Literal

# Ensure 'tiktoken' name is always bound and typed as Any for type-checkers
tiktoken: Any = None
try:
    import tiktoken as _tiktoken

    tiktoken = _tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False

from ..config.logging import LogLevel, StructuredLogger, create_logger  # noqa: E402

# ============================================================================
# Configuration and Enums
# ============================================================================


class PruningStrategy(str, Enum):
    """Pruning strategy enumeration."""

    RECENCY = "recency"
    IMPORTANCE = "importance"
    SLIDING_WINDOW = "sliding_window"
    RELEVANCE = "relevance"
    HYBRID = "hybrid"


class ImportanceLevel(str, Enum):
    """Message importance level."""

    CRITICAL = "critical"  # Errors, decisions, code changes
    HIGH = "high"  # Tool results, important responses
    MEDIUM = "medium"  # Regular conversation
    LOW = "low"  # Routine messages


@dataclass
class ContextConfig:
    """
    Configuration for context management.

    Attributes:
        max_tokens: Maximum context tokens (default: 100,000)
        pruning_threshold: Trigger pruning at this % of max (default: 0.8)
        min_tokens_to_keep: Minimum tokens to preserve (default: 10,000)
        default_strategy: Default pruning strategy
        preserve_recent_count: Number of recent messages to always keep
        enable_summarization: Whether to enable summarization
        summarization_threshold: Trigger summarization at this % (default: 0.6)
        relevance_cache_size: Max cached relevance scores
        model_name: Model name for token counting (default: "gpt-4")
    """

    max_tokens: int = 100_000
    pruning_threshold: float = 0.8
    min_tokens_to_keep: int = 10_000
    default_strategy: PruningStrategy = PruningStrategy.HYBRID
    preserve_recent_count: int = 10
    enable_summarization: bool = True
    summarization_threshold: float = 0.6
    relevance_cache_size: int = 1000
    model_name: str = "gpt-4"


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ContextMetadata:
    """
    Metadata for a context segment (message or group of messages).

    Attributes:
        timestamp: When the segment was created
        importance: Importance level
        token_count: Number of tokens in segment
        topics: Extracted topics/keywords
        message_type: Type of message (user, assistant, system, tool)
        has_code: Whether segment contains code
        has_error: Whether segment contains error
        relevance_score: Cached relevance score (if computed)
        is_summarized: Whether this segment has been summarized
    """

    timestamp: datetime = field(default_factory=datetime.now)
    importance: ImportanceLevel = ImportanceLevel.MEDIUM
    token_count: int = 0
    topics: list[str] = field(default_factory=list)
    message_type: str = "unknown"
    has_code: bool = False
    has_error: bool = False
    relevance_score: float | None = None
    is_summarized: bool = False


@dataclass
class ContextSegment:
    """
    A segment of conversation context with metadata.

    Attributes:
        content: The actual message content
        metadata: Associated metadata
        segment_id: Unique identifier
    """

    content: Any
    metadata: ContextMetadata
    segment_id: str = field(default_factory=lambda: f"seg_{int(time.time() * 1000)}")


@dataclass
class ContextCheckpoint:
    """
    Checkpoint for context state.

    Attributes:
        segments: Saved context segments
        timestamp: When checkpoint was created
        total_tokens: Total token count at checkpoint
        checkpoint_id: Unique identifier
    """

    segments: list[ContextSegment]
    timestamp: datetime = field(default_factory=datetime.now)
    total_tokens: int = 0
    checkpoint_id: str = field(default_factory=lambda: f"ckpt_{int(time.time() * 1000)}")


# ============================================================================
# Token Counting
# ============================================================================


class TokenCounter:
    """
    Token counter with tiktoken integration and fallback.

    Provides accurate token counting for different models with graceful
    degradation if tiktoken is not available.
    """

    def __init__(self, model_name: str = "gpt-4", logger: StructuredLogger | None = None):
        """
        Initialize token counter.

        Args:
            model_name: Model name for encoding selection
            logger: Optional logger instance
        """
        self.model_name = model_name
        self.logger = logger or create_logger("token_counter", LogLevel.INFO)
        # allow Encoding or None without type conflict
        self._encoding: Any | None = None
        self._initialize_encoding()

    def _initialize_encoding(self) -> None:
        """Initialize tiktoken encoding."""
        if not TIKTOKEN_AVAILABLE:
            self.logger.warning("tiktoken not available, using fallback estimation", extra={"model": self.model_name})
            return

        try:
            self._encoding = tiktoken.encoding_for_model(self.model_name)
            encoding_name = getattr(self._encoding, "name", "unknown")
            self.logger.info(
                "Initialized tiktoken encoding", extra={"model": self.model_name, "encoding": encoding_name}
            )
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self._encoding = tiktoken.get_encoding("cl100k_base")
            self.logger.warning("Model not found, using cl100k_base encoding", extra={"model": self.model_name})

    def count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count
        """
        if self._encoding is not None:
            return len(self._encoding.encode(text))

        # Fallback: estimate 1 token ≈ 4 characters
        return len(text) // 4

    def count_message_tokens(self, message: Any) -> int:
        """
        Count tokens in a message object.

        Args:
            message: Message object (dict, string, or other)

        Returns:
            Token count
        """
        if isinstance(message, str):
            return self.count_tokens(message)

        if isinstance(message, dict):
            # Count all string values in dict
            total = 0
            for value in message.values():
                if isinstance(value, str):
                    total += self.count_tokens(value)
                elif isinstance(value, (list, dict)):
                    total += self.count_message_tokens(value)
            return total

        if isinstance(message, list):
            return sum(self.count_message_tokens(item) for item in message)

        # Fallback: convert to string
        return self.count_tokens(str(message))


# ============================================================================
# Base Pruner
# ============================================================================


class ContextPruner(ABC):
    """
    Abstract base class for context pruning strategies.

    Subclasses implement specific pruning algorithms.
    """

    def __init__(self, config: ContextConfig, token_counter: TokenCounter, logger: StructuredLogger | None = None):
        """
        Initialize pruner.

        Args:
            config: Context configuration
            token_counter: Token counter instance
            logger: Optional logger instance
        """
        self.config = config
        self.token_counter = token_counter
        self.logger = logger or create_logger("context_pruner", LogLevel.INFO)

    @abstractmethod
    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune segments to target token count.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments
        """
        pass

    def _calculate_total_tokens(self, segments: list[ContextSegment]) -> int:
        """Calculate total tokens in segments."""
        return sum(seg.metadata.token_count for seg in segments)

    def _is_message_pair(self, seg1: ContextSegment, seg2: ContextSegment) -> bool:
        """
        Check if two segments form a message pair (user + assistant).

        Args:
            seg1: First segment
            seg2: Second segment

        Returns:
            True if segments form a pair
        """
        return seg1.metadata.message_type == "user" and seg2.metadata.message_type == "assistant"


# ============================================================================
# Pruning Strategies
# ============================================================================


class RecencyBasedPruner(ContextPruner):
    """
    Prune based on recency - keep most recent messages.

    Removes oldest messages first while preserving message pairs.
    """

    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune segments keeping most recent ones.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments (most recent)
        """
        if not segments:
            return []

        current_tokens = self._calculate_total_tokens(segments)

        if current_tokens <= target_tokens:
            return segments

        self.logger.info(
            "Starting recency-based pruning",
            extra={"current_tokens": current_tokens, "target_tokens": target_tokens, "segments_count": len(segments)},
        )

        # Keep segments from the end, removing from the beginning
        pruned = segments.copy()
        tokens_removed = 0

        while pruned and self._calculate_total_tokens(pruned) > target_tokens:
            # Check if first two segments form a pair
            if len(pruned) >= 2 and self._is_message_pair(pruned[0], pruned[1]):
                # Remove both to preserve coherence
                removed_segments = pruned[:2]
                pruned = pruned[2:]
                tokens_removed += sum(seg.metadata.token_count for seg in removed_segments)
            else:
                # Remove single segment
                removed_segment = pruned[0]
                pruned = pruned[1:]
                tokens_removed += removed_segment.metadata.token_count

        self.logger.info(
            "Recency-based pruning complete",
            extra={
                "tokens_removed": tokens_removed,
                "segments_removed": len(segments) - len(pruned),
                "final_tokens": self._calculate_total_tokens(pruned),
            },
        )

        return pruned


class ImportanceBasedPruner(ContextPruner):
    """
    Prune based on importance - preserve critical messages.

    Keeps errors, decisions, code changes, and high-importance messages.
    """

    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune segments preserving important ones.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments (important ones preserved)
        """
        if not segments:
            return []

        current_tokens = self._calculate_total_tokens(segments)

        if current_tokens <= target_tokens:
            return segments

        self.logger.info(
            "Starting importance-based pruning",
            extra={"current_tokens": current_tokens, "target_tokens": target_tokens, "segments_count": len(segments)},
        )

        # Separate segments by importance
        critical = []
        high = []
        medium = []
        low = []

        for seg in segments:
            if seg.metadata.importance == ImportanceLevel.CRITICAL:
                critical.append(seg)
            elif seg.metadata.importance == ImportanceLevel.HIGH:
                high.append(seg)
            elif seg.metadata.importance == ImportanceLevel.MEDIUM:
                medium.append(seg)
            else:
                low.append(seg)

        # Build result starting with critical, then high, etc.
        result = critical.copy()
        result_tokens = self._calculate_total_tokens(result)

        # Add high importance if space allows
        for seg in high:
            if result_tokens + seg.metadata.token_count <= target_tokens:
                result.append(seg)
                result_tokens += seg.metadata.token_count

        # Add medium importance if space allows
        for seg in medium:
            if result_tokens + seg.metadata.token_count <= target_tokens:
                result.append(seg)
                result_tokens += seg.metadata.token_count

        # Add low importance if space allows
        for seg in low:
            if result_tokens + seg.metadata.token_count <= target_tokens:
                result.append(seg)
                result_tokens += seg.metadata.token_count

        # Sort by original order (timestamp)
        result.sort(key=lambda s: s.metadata.timestamp)

        self.logger.info(
            "Importance-based pruning complete",
            extra={
                "tokens_removed": current_tokens - result_tokens,
                "segments_removed": len(segments) - len(result),
                "final_tokens": result_tokens,
                "critical_kept": len([s for s in result if s.metadata.importance == ImportanceLevel.CRITICAL]),
                "high_kept": len([s for s in result if s.metadata.importance == ImportanceLevel.HIGH]),
            },
        )

        return result


class SlidingWindowPruner(ContextPruner):
    """
    Sliding window pruner - maintain fixed-size context window.

    Keeps a window of recent messages with smart boundaries.
    """

    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune segments using sliding window.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments (sliding window)
        """
        if not segments:
            return []

        current_tokens = self._calculate_total_tokens(segments)

        if current_tokens <= target_tokens:
            return segments

        self.logger.info(
            "Starting sliding window pruning",
            extra={"current_tokens": current_tokens, "target_tokens": target_tokens, "segments_count": len(segments)},
        )

        # Keep most recent segments that fit in window
        window: list[ContextSegment] = []
        window_tokens = 0

        # Start from the end (most recent)
        for seg in reversed(segments):
            if window_tokens + seg.metadata.token_count <= target_tokens:
                window.insert(0, seg)
                window_tokens += seg.metadata.token_count
            else:
                break

        # Ensure we keep at least preserve_recent_count messages
        if len(window) < self.config.preserve_recent_count:
            window = segments[-self.config.preserve_recent_count :]
            window_tokens = self._calculate_total_tokens(window)

        self.logger.info(
            "Sliding window pruning complete",
            extra={
                "tokens_removed": current_tokens - window_tokens,
                "segments_removed": len(segments) - len(window),
                "final_tokens": window_tokens,
            },
        )

        return window


# ============================================================================
# Relevance Scoring
# ============================================================================


class RelevanceScorer:
    """
    Relevance scorer using keyword-based similarity.

    Provides simple TF-IDF style relevance scoring without external dependencies.
    Can be upgraded to use embeddings (sentence-transformers) later.
    """

    def __init__(self, cache_size: int = 1000, logger: StructuredLogger | None = None):
        """
        Initialize relevance scorer.

        Args:
            cache_size: Maximum cached scores
            logger: Optional logger instance
        """
        self.cache_size = cache_size
        self.logger = logger or create_logger("relevance_scorer", LogLevel.INFO)
        self._score_cache: dict[tuple[str, str], float] = {}

    def extract_keywords(self, text: str, top_n: int = 10) -> list[str]:
        """
        Extract keywords from text.

        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return

        Returns:
            List of keywords
        """
        # Simple keyword extraction: lowercase, remove punctuation, count
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())

        # Filter common stop words
        stop_words = {
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
            "day",
            "get",
            "has",
            "him",
            "his",
            "how",
            "man",
            "new",
            "now",
            "old",
            "see",
            "two",
            "way",
            "who",
            "boy",
            "did",
            "its",
            "let",
            "put",
            "say",
            "she",
            "too",
            "use",
            "this",
            "that",
            "with",
            "from",
            "have",
            "they",
            "will",
            "what",
            "when",
            "your",
        }

        filtered = [w for w in words if w not in stop_words]

        # Count frequencies
        counter = Counter(filtered)

        return [word for word, _ in counter.most_common(top_n)]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts.

        Uses Jaccard similarity on keywords.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Check cache
        cache_key = (text1[:100], text2[:100])  # Use prefix for cache key
        if cache_key in self._score_cache:
            return self._score_cache[cache_key]

        # Extract keywords
        keywords1 = set(self.extract_keywords(text1))
        keywords2 = set(self.extract_keywords(text2))

        if not keywords1 or not keywords2:
            return 0.0

        # Jaccard similarity
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)

        similarity = intersection / union if union > 0 else 0.0

        # Cache result
        if len(self._score_cache) < self.cache_size:
            self._score_cache[cache_key] = similarity

        return similarity

    def score_relevance(self, segment: ContextSegment, query: str, boost_recent: bool = True) -> float:
        """
        Score segment relevance to a query.

        Args:
            segment: Context segment to score
            query: Query text
            boost_recent: Whether to boost recent segments

        Returns:
            Relevance score (0.0 to 1.0)
        """
        # Extract text from segment
        if isinstance(segment.content, str):
            text = segment.content
        elif isinstance(segment.content, dict):
            text = str(segment.content)
        else:
            text = str(segment.content)

        # Calculate base similarity
        similarity = self.calculate_similarity(text, query)

        # Boost for importance
        importance_boost = {
            ImportanceLevel.CRITICAL: 1.5,
            ImportanceLevel.HIGH: 1.2,
            ImportanceLevel.MEDIUM: 1.0,
            ImportanceLevel.LOW: 0.8,
        }
        similarity *= importance_boost.get(segment.metadata.importance, 1.0)

        # Boost for recency (decay over time)
        if boost_recent:
            age_seconds = (datetime.now() - segment.metadata.timestamp).total_seconds()
            age_hours = age_seconds / 3600
            recency_boost = max(0.5, 1.0 - (age_hours / 24))  # Decay over 24 hours
            similarity *= recency_boost

        # Boost for code/errors
        if segment.metadata.has_code:
            similarity *= 1.1
        if segment.metadata.has_error:
            similarity *= 1.3

        # Clamp to [0, 1]
        return min(1.0, similarity)

    def clear_cache(self) -> None:
        """Clear relevance score cache."""
        self._score_cache.clear()
        self.logger.info("Relevance score cache cleared")


class RelevanceBasedPruner(ContextPruner):
    """
    Prune based on relevance to current context.

    Keeps segments most relevant to recent conversation.
    """

    def __init__(
        self,
        config: ContextConfig,
        token_counter: TokenCounter,
        relevance_scorer: RelevanceScorer,
        logger: StructuredLogger | None = None,
    ):
        """
        Initialize relevance-based pruner.

        Args:
            config: Context configuration
            token_counter: Token counter instance
            relevance_scorer: Relevance scorer instance
            logger: Optional logger instance
        """
        super().__init__(config, token_counter, logger)
        self.relevance_scorer = relevance_scorer

    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune segments based on relevance.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments (most relevant)
        """
        if not segments:
            return []

        current_tokens = self._calculate_total_tokens(segments)

        if current_tokens <= target_tokens:
            return segments

        self.logger.info(
            "Starting relevance-based pruning",
            extra={"current_tokens": current_tokens, "target_tokens": target_tokens, "segments_count": len(segments)},
        )

        # Use recent segments as query context
        recent_segments = segments[-self.config.preserve_recent_count :]
        query_text = " ".join(str(seg.content) for seg in recent_segments)

        # Score all segments
        scored_segments = []
        for seg in segments:
            score = self.relevance_scorer.score_relevance(seg, query_text)
            seg.metadata.relevance_score = score
            scored_segments.append((score, seg))

        # Sort by relevance (descending)
        scored_segments.sort(key=lambda x: x[0], reverse=True)

        # Keep highest scoring segments that fit in target
        result = []
        result_tokens = 0

        for _score, seg in scored_segments:
            if result_tokens + seg.metadata.token_count <= target_tokens:
                result.append(seg)
                result_tokens += seg.metadata.token_count

        # Sort by original order (timestamp)
        result.sort(key=lambda s: s.metadata.timestamp)

        self.logger.info(
            "Relevance-based pruning complete",
            extra={
                "tokens_removed": current_tokens - result_tokens,
                "segments_removed": len(segments) - len(result),
                "final_tokens": result_tokens,
                "avg_relevance": sum(s.metadata.relevance_score or 0 for s in result) / len(result) if result else 0,
            },
        )

        return result


class HybridPruner(ContextPruner):
    """
    Hybrid pruner combining multiple strategies.

    Uses importance to preserve critical messages, then relevance
    for remaining content.
    """

    def __init__(
        self,
        config: ContextConfig,
        token_counter: TokenCounter,
        relevance_scorer: RelevanceScorer,
        logger: StructuredLogger | None = None,
    ):
        """
        Initialize hybrid pruner.

        Args:
            config: Context configuration
            token_counter: Token counter instance
            relevance_scorer: Relevance scorer instance
            logger: Optional logger instance
        """
        super().__init__(config, token_counter, logger)
        self.relevance_scorer = relevance_scorer

        # Initialize sub-pruners
        self.importance_pruner = ImportanceBasedPruner(config, token_counter, logger)
        self.relevance_pruner = RelevanceBasedPruner(config, token_counter, relevance_scorer, logger)

    def prune(self, segments: list[ContextSegment], target_tokens: int) -> list[ContextSegment]:
        """
        Prune using hybrid strategy.

        Args:
            segments: Context segments to prune
            target_tokens: Target token count after pruning

        Returns:
            Pruned segments
        """
        if not segments:
            return []

        current_tokens = self._calculate_total_tokens(segments)

        if current_tokens <= target_tokens:
            return segments

        self.logger.info(
            "Starting hybrid pruning",
            extra={"current_tokens": current_tokens, "target_tokens": target_tokens, "segments_count": len(segments)},
        )

        # First pass: Use importance to preserve critical/high priority
        # Target 70% of final target for this pass
        importance_target = int(target_tokens * 0.7)
        result = self.importance_pruner.prune(segments, importance_target)

        # Second pass: Use relevance on remaining if still over target
        result_tokens = self._calculate_total_tokens(result)
        if result_tokens > target_tokens:
            result = self.relevance_pruner.prune(result, target_tokens)

        final_tokens = self._calculate_total_tokens(result)

        self.logger.info(
            "Hybrid pruning complete",
            extra={
                "tokens_removed": current_tokens - final_tokens,
                "segments_removed": len(segments) - len(result),
                "final_tokens": final_tokens,
            },
        )

        return result


# ============================================================================
# Context Summarization
# ============================================================================


@dataclass
class SummarySegment:
    """
    A summarized segment of conversation.

    Attributes:
        summary: Summary text
        original_segments: Original segment IDs that were summarized
        token_count: Tokens in summary
        timestamp: When summary was created
    """

    summary: str
    original_segments: list[str]
    token_count: int
    timestamp: datetime = field(default_factory=datetime.now)


class ContextSummarizer:
    """
    Context summarizer for conversation history.

    Implements incremental summarization to compress older context.
    Note: Actual LLM-based summarization would require agent integration.
    This provides the infrastructure and placeholder implementation.
    """

    def __init__(self, token_counter: TokenCounter, logger: StructuredLogger | None = None):
        """
        Initialize context summarizer.

        Args:
            token_counter: Token counter instance
            logger: Optional logger instance
        """
        self.token_counter = token_counter
        self.logger = logger or create_logger("context_summarizer", LogLevel.INFO)
        self.summarized_segments: list[SummarySegment] = []

    def should_summarize(self, segments: list[ContextSegment], threshold_tokens: int) -> bool:
        """
        Check if summarization should be triggered.

        Args:
            segments: Context segments
            threshold_tokens: Token threshold for summarization

        Returns:
            True if summarization should occur
        """
        total_tokens = sum(seg.metadata.token_count for seg in segments)
        return total_tokens > threshold_tokens

    def create_summary(self, segments: list[ContextSegment], max_summary_tokens: int = 500) -> SummarySegment:
        """
        Create summary of segments.

        This is a placeholder implementation. In production, this would
        use an LLM to generate intelligent summaries.

        Args:
            segments: Segments to summarize
            max_summary_tokens: Maximum tokens in summary

        Returns:
            Summary segment
        """
        self.logger.info(
            "Creating summary", extra={"segments_count": len(segments), "max_summary_tokens": max_summary_tokens}
        )

        # Placeholder: Extract key information
        summary_parts = []

        # Count message types
        user_msgs = sum(1 for s in segments if s.metadata.message_type == "user")
        assistant_msgs = sum(1 for s in segments if s.metadata.message_type == "assistant")

        summary_parts.append(f"[Summary of {len(segments)} messages: {user_msgs} user, {assistant_msgs} assistant]")

        # Extract errors
        errors = [s for s in segments if s.metadata.has_error]
        if errors:
            summary_parts.append(f"Errors encountered: {len(errors)}")

        # Extract code segments
        code_segments = [s for s in segments if s.metadata.has_code]
        if code_segments:
            summary_parts.append(f"Code segments: {len(code_segments)}")

        # Extract topics
        all_topics = []
        for seg in segments:
            all_topics.extend(seg.metadata.topics)

        if all_topics:
            topic_counts = Counter(all_topics)
            top_topics = [topic for topic, _ in topic_counts.most_common(3)]
            summary_parts.append(f"Topics: {', '.join(top_topics)}")

        summary_text = " | ".join(summary_parts)

        # Create summary segment
        summary = SummarySegment(
            summary=summary_text,
            original_segments=[seg.segment_id for seg in segments],
            token_count=self.token_counter.count_tokens(summary_text),
        )

        self.summarized_segments.append(summary)

        self.logger.info(
            "Summary created",
            extra={
                "summary_tokens": summary.token_count,
                "original_tokens": sum(s.metadata.token_count for s in segments),
                "compression_ratio": summary.token_count / sum(s.metadata.token_count for s in segments),
            },
        )

        return summary

    def inject_summary(self, segments: list[ContextSegment], summary: SummarySegment) -> list[ContextSegment]:
        """
        Inject summary into segment list, replacing original segments.

        Args:
            segments: Original segments
            summary: Summary to inject

        Returns:
            Segments with summary injected
        """
        # Find segments that were summarized
        summarized_ids = set(summary.original_segments)

        # Keep segments not in summary
        remaining = [s for s in segments if s.segment_id not in summarized_ids]

        # Create summary as a context segment
        summary_segment = ContextSegment(
            content=summary.summary,
            metadata=ContextMetadata(
                timestamp=summary.timestamp,
                importance=ImportanceLevel.HIGH,
                token_count=summary.token_count,
                message_type="summary",
                is_summarized=True,
            ),
        )

        # Insert summary at the beginning
        result = [summary_segment] + remaining

        self.logger.info(
            "Summary injected",
            extra={
                "segments_before": len(segments),
                "segments_after": len(result),
                "segments_summarized": len(summarized_ids),
            },
        )

        return result


# ============================================================================
# Context Manager - Main Orchestration
# ============================================================================


class ContextManager:
    """
    Main context manager orchestrating all context operations.

    Monitors token usage, triggers pruning/summarization, manages checkpoints,
    and provides context health metrics.
    """

    def __init__(self, config: ContextConfig | None = None, logger: StructuredLogger | None = None):
        """
        Initialize context manager.

        Args:
            config: Context configuration (uses defaults if None)
            logger: Optional logger instance
        """
        self.config = config or ContextConfig()
        self.logger = logger or create_logger("context_manager", LogLevel.INFO)

        # Initialize components
        self.token_counter = TokenCounter(self.config.model_name, self.logger)
        self.relevance_scorer = RelevanceScorer(self.config.relevance_cache_size, self.logger)
        self.summarizer = ContextSummarizer(self.token_counter, self.logger)

        # Initialize pruners
        self.pruners: dict[PruningStrategy, ContextPruner] = {
            PruningStrategy.RECENCY: RecencyBasedPruner(self.config, self.token_counter, self.logger),
            PruningStrategy.IMPORTANCE: ImportanceBasedPruner(self.config, self.token_counter, self.logger),
            PruningStrategy.SLIDING_WINDOW: SlidingWindowPruner(self.config, self.token_counter, self.logger),
            PruningStrategy.RELEVANCE: RelevanceBasedPruner(
                self.config, self.token_counter, self.relevance_scorer, self.logger
            ),
            PruningStrategy.HYBRID: HybridPruner(self.config, self.token_counter, self.relevance_scorer, self.logger),
        }

        # State
        self.segments: list[ContextSegment] = []
        self.checkpoints: list[ContextCheckpoint] = []
        self.total_tokens = 0
        self.pruning_count = 0
        self.summarization_count = 0

    def add_message(
        self, content: Any, message_type: str = "unknown", importance: ImportanceLevel = ImportanceLevel.MEDIUM
    ) -> ContextSegment:
        """
        Add a message to context.

        Args:
            content: Message content
            message_type: Type of message (user, assistant, system, tool)
            importance: Importance level

        Returns:
            Created context segment
        """
        # Count tokens
        token_count = self.token_counter.count_message_tokens(content)

        # Extract metadata
        content_str = str(content).lower()
        has_code = any(marker in content_str for marker in ["```", "def ", "class ", "import "])
        has_error = any(marker in content_str for marker in ["error", "exception", "failed", "traceback"])

        # Extract topics (simple keyword extraction)
        topics = self.relevance_scorer.extract_keywords(content_str, top_n=5)

        # Determine importance
        if has_error:
            importance = ImportanceLevel.CRITICAL
        elif has_code:
            importance = max(importance, ImportanceLevel.HIGH)

        # Create segment
        metadata = ContextMetadata(
            importance=importance,
            token_count=token_count,
            topics=topics,
            message_type=message_type,
            has_code=has_code,
            has_error=has_error,
        )

        segment = ContextSegment(content=content, metadata=metadata)

        self.segments.append(segment)
        self.total_tokens += token_count

        self.logger.debug(
            "Message added to context",
            extra={
                "segment_id": segment.segment_id,
                "message_type": message_type,
                "importance": importance.value,
                "token_count": token_count,
                "total_tokens": self.total_tokens,
            },
        )

        # Check if pruning/summarization needed
        self._auto_manage_context()

        return segment

    def _auto_manage_context(self) -> None:
        """Automatically manage context based on thresholds."""
        pruning_threshold_tokens = int(self.config.max_tokens * self.config.pruning_threshold)
        summarization_threshold_tokens = int(self.config.max_tokens * self.config.summarization_threshold)

        # Check if summarization needed
        if (
            self.config.enable_summarization
            and self.total_tokens > summarization_threshold_tokens
            and len(self.segments) > self.config.preserve_recent_count * 2
        ):
            self._trigger_summarization()

        # Check if pruning needed
        if self.total_tokens > pruning_threshold_tokens:
            self._trigger_pruning()

    def _trigger_summarization(self) -> None:
        """Trigger automatic summarization."""
        self.logger.info(
            "Triggering automatic summarization",
            extra={"total_tokens": self.total_tokens, "segments_count": len(self.segments)},
        )

        # Summarize older segments (keep recent ones)
        segments_to_summarize = self.segments[: -self.config.preserve_recent_count]

        if len(segments_to_summarize) < 5:
            return  # Not enough to summarize

        # Create summary
        summary = self.summarizer.create_summary(segments_to_summarize)

        # Inject summary
        self.segments = self.summarizer.inject_summary(self.segments, summary)

        # Recalculate total tokens
        self.total_tokens = sum(seg.metadata.token_count for seg in self.segments)
        self.summarization_count += 1

        self.logger.info(
            "Summarization complete",
            extra={
                "new_total_tokens": self.total_tokens,
                "new_segments_count": len(self.segments),
                "summarization_count": self.summarization_count,
            },
        )

    def _trigger_pruning(self) -> None:
        """Trigger automatic pruning."""
        target_tokens = self.config.min_tokens_to_keep

        self.logger.info(
            "Triggering automatic pruning",
            extra={
                "total_tokens": self.total_tokens,
                "target_tokens": target_tokens,
                "strategy": self.config.default_strategy.value,
            },
        )

        # Get pruner
        pruner = self.pruners.get(self.config.default_strategy)

        if pruner is None:
            self.logger.error(f"Unknown pruning strategy: {self.config.default_strategy}")
            return

        # Prune
        self.segments = pruner.prune(self.segments, target_tokens)

        # Recalculate total tokens
        self.total_tokens = sum(seg.metadata.token_count for seg in self.segments)
        self.pruning_count += 1

        self.logger.info(
            "Pruning complete",
            extra={
                "new_total_tokens": self.total_tokens,
                "new_segments_count": len(self.segments),
                "pruning_count": self.pruning_count,
            },
        )

    def manual_prune(self, strategy: PruningStrategy, target_tokens: int | None = None) -> None:
        """
        Manually trigger pruning with specific strategy.

        Args:
            strategy: Pruning strategy to use
            target_tokens: Target token count (uses config default if None)
        """
        target = target_tokens or self.config.min_tokens_to_keep

        pruner = self.pruners.get(strategy)
        if pruner is None:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        self.logger.info("Manual pruning triggered", extra={"strategy": strategy.value, "target_tokens": target})

        self.segments = pruner.prune(self.segments, target)
        self.total_tokens = sum(seg.metadata.token_count for seg in self.segments)
        self.pruning_count += 1

    def create_checkpoint(self) -> ContextCheckpoint:
        """
        Create a checkpoint of current context state.

        Returns:
            Created checkpoint
        """
        checkpoint = ContextCheckpoint(segments=self.segments.copy(), total_tokens=self.total_tokens)

        self.checkpoints.append(checkpoint)

        self.logger.info(
            "Checkpoint created",
            extra={
                "checkpoint_id": checkpoint.checkpoint_id,
                "segments_count": len(checkpoint.segments),
                "total_tokens": checkpoint.total_tokens,
            },
        )

        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Restore context from a checkpoint.

        Args:
            checkpoint_id: Checkpoint ID to restore

        Returns:
            True if restored successfully
        """
        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id == checkpoint_id:
                self.segments = checkpoint.segments.copy()
                self.total_tokens = checkpoint.total_tokens

                self.logger.info(
                    "Checkpoint restored",
                    extra={
                        "checkpoint_id": checkpoint_id,
                        "segments_count": len(self.segments),
                        "total_tokens": self.total_tokens,
                    },
                )

                return True

        self.logger.warning(f"Checkpoint not found: {checkpoint_id}")
        return False

    def get_statistics(self) -> dict[str, Any]:
        """
        Get context statistics and health metrics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_tokens": self.total_tokens,
            "max_tokens": self.config.max_tokens,
            "token_usage_percent": (self.total_tokens / self.config.max_tokens) * 100,
            "segments_count": len(self.segments),
            "pruning_count": self.pruning_count,
            "summarization_count": self.summarization_count,
            "checkpoints_count": len(self.checkpoints),
            "importance_distribution": {
                "critical": len([s for s in self.segments if s.metadata.importance == ImportanceLevel.CRITICAL]),
                "high": len([s for s in self.segments if s.metadata.importance == ImportanceLevel.HIGH]),
                "medium": len([s for s in self.segments if s.metadata.importance == ImportanceLevel.MEDIUM]),
                "low": len([s for s in self.segments if s.metadata.importance == ImportanceLevel.LOW]),
            },
            "message_types": {
                msg_type: len([s for s in self.segments if s.metadata.message_type == msg_type])
                for msg_type in {s.metadata.message_type for s in self.segments}
            },
            "has_code_count": len([s for s in self.segments if s.metadata.has_code]),
            "has_error_count": len([s for s in self.segments if s.metadata.has_error]),
            "summarized_count": len([s for s in self.segments if s.metadata.is_summarized]),
        }

    def get_health_status(self) -> Literal["healthy", "warning", "critical"]:
        """
        Get context health status.

        Returns:
            Health status
        """
        usage_percent = (self.total_tokens / self.config.max_tokens) * 100

        if usage_percent >= 90:
            return "critical"
        if usage_percent >= 70:
            return "warning"
        return "healthy"

    def clear_context(self) -> None:
        """Clear all context segments."""
        self.segments.clear()
        self.total_tokens = 0

        self.logger.info("Context cleared")

    def get_segments(self) -> list[ContextSegment]:
        """Get all context segments."""
        return self.segments.copy()

    def get_messages(self) -> list[Any]:
        """Get all message contents."""
        return [seg.content for seg in self.segments]


# ============================================================================
# Utility Functions
# ============================================================================


def create_context_manager(
    max_tokens: int = 100_000,
    model_name: str = "gpt-4",
    strategy: PruningStrategy = PruningStrategy.HYBRID,
    enable_summarization: bool = True,
    logger: StructuredLogger | None = None,
) -> ContextManager:
    """
    Create a context manager with custom configuration.

    Args:
        max_tokens: Maximum context tokens
        model_name: Model name for token counting
        strategy: Default pruning strategy
        enable_summarization: Whether to enable summarization
        logger: Optional logger instance

    Returns:
        Configured context manager
    """
    config = ContextConfig(
        max_tokens=max_tokens,
        model_name=model_name,
        default_strategy=strategy,
        enable_summarization=enable_summarization,
    )

    return ContextManager(config, logger)


__all__ = [
    # Configuration
    "ContextConfig",
    "PruningStrategy",
    "ImportanceLevel",
    # Data Models
    "ContextMetadata",
    "ContextSegment",
    "ContextCheckpoint",
    "SummarySegment",
    # Core Components
    "TokenCounter",
    "RelevanceScorer",
    "ContextSummarizer",
    # Pruners
    "ContextPruner",
    "RecencyBasedPruner",
    "ImportanceBasedPruner",
    "SlidingWindowPruner",
    "RelevanceBasedPruner",
    # Main Manager
    "ContextManager",
    # Utilities
    "create_context_manager",
    "TIKTOKEN_AVAILABLE",
]

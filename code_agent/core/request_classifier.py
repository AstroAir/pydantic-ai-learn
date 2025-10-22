"""
Request Classification System

Analyzes and classifies user requests for intelligent routing.

Features:
- Difficulty level assessment (simple, moderate, complex)
- Request mode detection (chat vs agent)
- Heuristic-based classification
- Optional LLM-based classification
- Confidence scoring
- Result caching

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import hashlib
import re
import time
from typing import TYPE_CHECKING

from .routing_types import DifficultyLevel, RequestClassification, RequestMode

if TYPE_CHECKING:
    from ..config.logging import StructuredLogger
    from .routing_config import ClassificationConfig


class RequestClassifier:
    """
    Request classification system for intelligent routing.

    Classifies requests by:
    - Difficulty level (simple, moderate, complex)
    - Request mode (chat vs agent)
    - Tool requirements
    - Estimated token usage
    """

    def __init__(
        self,
        config: ClassificationConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Initialize request classifier.

        Args:
            config: Classification configuration
            logger: Optional logger
        """
        from .routing_config import ClassificationConfig

        self.config = config or ClassificationConfig()
        self.logger = logger
        self._cache: dict[str, tuple[RequestClassification, float]] = {}

    def classify(self, prompt: str) -> RequestClassification:
        """
        Classify a user request.

        Args:
            prompt: User prompt to classify

        Returns:
            Classification result
        """
        if not self.config.enabled:
            # Return default classification
            return RequestClassification(
                difficulty=DifficultyLevel.MODERATE,
                mode=RequestMode.AGENT,
                confidence=0.5,
                reasoning="Classification disabled",
            )

        # Check cache
        cache_key = self._get_cache_key(prompt)
        if cache_key in self._cache:
            cached_result, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self.config.cache_ttl:
                if self.logger:
                    self.logger.debug("Using cached classification", extra={"cache_key": cache_key})
                return cached_result

        # Perform classification
        result = self._classify_llm_based(prompt) if self.config.use_llm else self._classify_heuristic(prompt)

        # Cache result
        self._cache[cache_key] = (result, time.time())

        # Log classification
        if self.logger:
            self.logger.info(
                "Request classified",
                extra={
                    "difficulty": result.difficulty.value,
                    "mode": result.mode.value,
                    "confidence": result.confidence,
                    "requires_tools": result.requires_tools,
                },
            )

        return result

    def _classify_heuristic(self, prompt: str) -> RequestClassification:
        """
        Classify request using heuristics.

        Args:
            prompt: User prompt

        Returns:
            Classification result
        """
        prompt_lower = prompt.lower()

        # Assess difficulty
        difficulty, difficulty_confidence = self._assess_difficulty(prompt_lower)

        # Detect mode
        mode, mode_confidence = self._detect_mode(prompt_lower)

        # Determine if tools are required
        requires_tools = self._requires_tools(prompt_lower)

        # Estimate tokens
        estimated_tokens = self._estimate_tokens(prompt)

        # Extract keywords
        keywords = self._extract_keywords(prompt_lower)

        # Overall confidence is average of difficulty and mode confidence
        confidence = (difficulty_confidence + mode_confidence) / 2

        # Generate reasoning
        reasoning = self._generate_reasoning(difficulty, mode, requires_tools, keywords)

        return RequestClassification(
            difficulty=difficulty,
            mode=mode,
            confidence=confidence,
            reasoning=reasoning,
            requires_tools=requires_tools,
            estimated_tokens=estimated_tokens,
            keywords=keywords,
        )

    def _classify_llm_based(self, prompt: str) -> RequestClassification:
        """
        Classify request using LLM.

        Args:
            prompt: User prompt

        Returns:
            Classification result
        """
        # TODO: Implement LLM-based classification
        if self.logger:
            self.logger.warning("LLM-based classification not yet implemented, falling back to heuristic")

        return self._classify_heuristic(prompt)

    # ========================================================================
    # Difficulty Assessment
    # ========================================================================

    def _assess_difficulty(self, prompt_lower: str) -> tuple[DifficultyLevel, float]:
        """
        Assess difficulty level of request.

        Args:
            prompt_lower: Lowercase prompt

        Returns:
            Tuple of (difficulty level, confidence)
        """
        simple_score = 0.0
        moderate_score = 0.0
        complex_score = 0.0

        # Check keywords
        for keyword in self.config.difficulty_keywords.get("simple", []):
            if keyword in prompt_lower:
                simple_score += 1.0

        for keyword in self.config.difficulty_keywords.get("moderate", []):
            if keyword in prompt_lower:
                moderate_score += 1.0

        for keyword in self.config.difficulty_keywords.get("complex", []):
            if keyword in prompt_lower:
                complex_score += 1.0

        # Check prompt length (longer prompts tend to be more complex)
        word_count = len(prompt_lower.split())
        if word_count < 10:
            simple_score += 0.5
        elif word_count < 30:
            moderate_score += 0.5
        else:
            complex_score += 0.5

        # Check for code patterns
        if self._has_code_patterns(prompt_lower):
            complex_score += 1.0

        # Check for multi-step indicators
        if self._is_multi_step(prompt_lower):
            complex_score += 1.0

        # Determine difficulty
        scores = {
            DifficultyLevel.SIMPLE: simple_score,
            DifficultyLevel.MODERATE: moderate_score,
            DifficultyLevel.COMPLEX: complex_score,
        }

        max_score = max(scores.values())
        if max_score == 0:
            # Default to moderate
            return DifficultyLevel.MODERATE, 0.5

        difficulty = max(scores, key=scores.get)  # type: ignore
        confidence = min(max_score / (sum(scores.values()) or 1), 1.0)

        return difficulty, confidence

    # ========================================================================
    # Mode Detection
    # ========================================================================

    def _detect_mode(self, prompt_lower: str) -> tuple[RequestMode, float]:
        """
        Detect request mode (chat vs agent).

        Args:
            prompt_lower: Lowercase prompt

        Returns:
            Tuple of (mode, confidence)
        """
        chat_score = 0.0
        agent_score = 0.0

        # Check keywords
        for keyword in self.config.mode_keywords.get("chat", []):
            if keyword in prompt_lower:
                chat_score += 1.0

        for keyword in self.config.mode_keywords.get("agent", []):
            if keyword in prompt_lower:
                agent_score += 1.0

        # Check for action verbs (agent mode)
        action_verbs = ["create", "generate", "refactor", "fix", "implement", "modify", "delete", "build"]
        for verb in action_verbs:
            if re.search(r"\b" + verb + r"\b", prompt_lower):
                agent_score += 1.0

        # Check for question words (chat mode)
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        for word in question_words:
            if re.search(r"\b" + word + r"\b", prompt_lower):
                chat_score += 0.5

        # Determine mode
        if agent_score > chat_score:
            mode = RequestMode.AGENT
            confidence = min(agent_score / (agent_score + chat_score), 1.0)
        elif chat_score > agent_score:
            mode = RequestMode.CHAT
            confidence = min(chat_score / (agent_score + chat_score), 1.0)
        else:
            # Default to agent mode
            mode = RequestMode.AGENT
            confidence = 0.5

        return mode, confidence

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _requires_tools(self, prompt_lower: str) -> bool:
        """Check if request requires tool usage."""
        tool_indicators = [
            "create",
            "generate",
            "refactor",
            "fix",
            "implement",
            "modify",
            "delete",
            "analyze",
            "file",
            "code",
        ]
        return any(indicator in prompt_lower for indicator in tool_indicators)

    def _estimate_tokens(self, prompt: str) -> int:
        """Estimate token usage (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(prompt) // 4

    def _extract_keywords(self, prompt_lower: str) -> list[str]:
        """Extract important keywords from prompt."""
        # Simple keyword extraction
        words = prompt_lower.split()
        # Filter out common words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:10]  # Return top 10

    def _has_code_patterns(self, prompt_lower: str) -> bool:
        """Check if prompt contains code-related patterns."""
        code_patterns = [
            r"function",
            r"class",
            r"method",
            r"variable",
            r"\.py",
            r"\.js",
            r"\.ts",
            r"import",
            r"def\s",
            r"async",
        ]
        return any(re.search(pattern, prompt_lower) for pattern in code_patterns)

    def _is_multi_step(self, prompt_lower: str) -> bool:
        """Check if prompt indicates multi-step operation."""
        multi_step_indicators = ["first", "then", "after", "finally", "step", "and then", "followed by"]
        return any(indicator in prompt_lower for indicator in multi_step_indicators)

    def _generate_reasoning(
        self,
        difficulty: DifficultyLevel,
        mode: RequestMode,
        requires_tools: bool,
        keywords: list[str],
    ) -> str:
        """Generate human-readable reasoning for classification."""
        parts = [
            f"Classified as {difficulty.value} difficulty",
            f"{mode.value} mode",
        ]

        if requires_tools:
            parts.append("requires tools")

        if keywords:
            parts.append(f"key terms: {', '.join(keywords[:3])}")

        return "; ".join(parts)

    def _get_cache_key(self, prompt: str) -> str:
        """Generate cache key for prompt."""
        return hashlib.md5(prompt.encode()).hexdigest()

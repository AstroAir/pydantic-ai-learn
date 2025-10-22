"""
Prompt Enhancement System

Automatically enhances and optimizes user prompts for better AI comprehension.

Features:
- Rule-based enhancement using heuristics
- LLM-based enhancement using lightweight models
- Hybrid enhancement combining both approaches
- Configurable enhancement strategies
- Confidence scoring

Author: The Augster
Python Version: 3.12+
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .routing_types import EnhancementStrategy, PromptEnhancement

if TYPE_CHECKING:
    from ..config.logging import StructuredLogger
    from .routing_config import EnhancementConfig


class PromptEnhancer:
    """
    Prompt enhancement system for optimizing user queries.

    Enhances prompts by:
    - Adding clarity and specificity
    - Structuring requests
    - Adding context when needed
    - Removing ambiguity
    """

    def __init__(
        self,
        config: EnhancementConfig | None = None,
        logger: StructuredLogger | None = None,
    ) -> None:
        """
        Initialize prompt enhancer.

        Args:
            config: Enhancement configuration
            logger: Optional logger
        """
        from .routing_config import EnhancementConfig

        self.config = config or EnhancementConfig()
        self.logger = logger

    def enhance(self, prompt: str) -> PromptEnhancement:
        """
        Enhance a user prompt.

        Args:
            prompt: Original user prompt

        Returns:
            Enhancement result with enhanced prompt and metadata
        """
        if not self.config.enabled:
            return PromptEnhancement(
                original_prompt=prompt,
                enhanced_prompt=prompt,
                strategy=EnhancementStrategy.NONE,
                confidence=0.0,  # Disabled means no confidence in enhancement
            )

        # Select enhancement strategy
        if self.config.strategy == EnhancementStrategy.RULE_BASED:
            return self._enhance_rule_based(prompt)
        if self.config.strategy == EnhancementStrategy.LLM_BASED:
            return self._enhance_llm_based(prompt)
        if self.config.strategy == EnhancementStrategy.HYBRID:
            return self._enhance_hybrid(prompt)
        return PromptEnhancement(
            original_prompt=prompt,
            enhanced_prompt=prompt,
            strategy=EnhancementStrategy.NONE,
            confidence=1.0,
        )

    def _enhance_rule_based(self, prompt: str) -> PromptEnhancement:
        """
        Enhance prompt using rule-based heuristics.

        Args:
            prompt: Original prompt

        Returns:
            Enhancement result
        """
        enhanced = prompt
        improvements: list[str] = []
        confidence = 0.8

        # Rule 1: Add context for vague pronouns
        if self._has_vague_pronouns(prompt):
            # Add a concise note to be more specific
            enhanced = f"{prompt} - be specific"
            improvements.append("Added specificity hint for vague pronouns")

        # Rule 2: Structure multi-part requests
        if self._is_multi_part_request(prompt):
            enhanced = self._structure_multi_part(prompt)
            if enhanced != prompt:
                improvements.append("Structured multi-part request")

        # Rule 3: Add specificity to code-related requests
        if self._is_code_request(prompt) and not self._has_file_context(prompt):
            enhanced = self._add_code_context_hint(enhanced)
            if enhanced != prompt:
                improvements.append("Added code context hint")

        # Rule 4: Clarify action verbs
        if self._has_ambiguous_verbs(prompt):
            enhanced = self._clarify_verbs(enhanced)
            if enhanced != prompt:
                improvements.append("Clarified action verbs")

        # Check length increase - only warn if exceeded, don't revert
        length_increase = (len(enhanced) - len(prompt)) / len(prompt) if len(prompt) > 0 else 0
        if length_increase > self.config.max_length_increase and improvements:
            # Reduce confidence but keep the enhancement
            confidence = max(0.5, confidence - 0.2)
            improvements.append(f"Enhancement increased length by {length_increase:.0%}")

        return PromptEnhancement(
            original_prompt=prompt,
            enhanced_prompt=enhanced,
            strategy=EnhancementStrategy.RULE_BASED,
            improvements=improvements,
            confidence=confidence,
        )

    def _enhance_llm_based(self, prompt: str) -> PromptEnhancement:
        """
        Enhance prompt using LLM.

        Args:
            prompt: Original prompt

        Returns:
            Enhancement result
        """
        # TODO: Implement LLM-based enhancement
        # This would use a lightweight model to enhance the prompt
        if self.logger:
            self.logger.warning("LLM-based enhancement not yet implemented, falling back to rule-based")

        return self._enhance_rule_based(prompt)

    def _enhance_hybrid(self, prompt: str) -> PromptEnhancement:
        """
        Enhance prompt using hybrid approach.

        Args:
            prompt: Original prompt

        Returns:
            Enhancement result
        """
        # Start with rule-based
        rule_result = self._enhance_rule_based(prompt)

        # If confidence is low, try LLM-based
        if rule_result.confidence < self.config.min_confidence:
            return self._enhance_llm_based(prompt)

        return rule_result

    # ========================================================================
    # Heuristic Helpers
    # ========================================================================

    def _has_vague_pronouns(self, prompt: str) -> bool:
        """Check if prompt has vague pronouns without clear antecedents."""
        vague_pronouns = ["it", "this", "that", "these", "those", "them"]
        words = prompt.lower().split()
        return any(pronoun in words for pronoun in vague_pronouns)

    def _is_multi_part_request(self, prompt: str) -> bool:
        """Check if prompt contains multiple requests."""
        # Look for conjunctions and multiple sentences
        has_and = " and " in prompt.lower()
        has_multiple_sentences = prompt.count(".") > 1 or prompt.count("?") > 1
        return has_and or has_multiple_sentences

    def _structure_multi_part(self, prompt: str) -> str:
        """Structure multi-part requests with numbering."""
        # Simple heuristic: if it has "and", split and number
        if " and " in prompt.lower():
            parts = re.split(r"\s+and\s+", prompt, flags=re.IGNORECASE)
            if len(parts) > 1:
                structured = "Please:\n"
                for i, part in enumerate(parts, 1):
                    structured += f"{i}. {part.strip()}\n"
                return structured.strip()
        return prompt

    def _is_code_request(self, prompt: str) -> bool:
        """Check if prompt is code-related."""
        code_keywords = [
            "code",
            "function",
            "class",
            "method",
            "refactor",
            "implement",
            "debug",
            "fix",
            "generate",
            "create",
            "analyze",
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in code_keywords)

    def _has_file_context(self, prompt: str) -> bool:
        """Check if prompt mentions specific files."""
        # Look for file extensions or path-like patterns
        file_patterns = [r"\.\w+", r"/\w+", r"\\\w+", r"\.py", r"\.js", r"\.ts"]
        return any(re.search(pattern, prompt) for pattern in file_patterns)

    def _add_code_context_hint(self, prompt: str) -> str:
        """Add hint to specify file context."""
        if not prompt.endswith((".", "?", "!")):
            prompt += "."
        return prompt + " (Please specify the file or code context if applicable)"

    def _has_ambiguous_verbs(self, prompt: str) -> bool:
        """Check for ambiguous action verbs."""
        ambiguous_verbs = ["fix", "improve", "update", "change", "modify"]
        prompt_lower = prompt.lower()
        return any(verb in prompt_lower for verb in ambiguous_verbs)

    def _clarify_verbs(self, prompt: str) -> str:
        """Clarify ambiguous verbs with more specific language."""
        # This is a simple example - in practice, this would be more sophisticated
        replacements = {
            "fix": "fix (identify and resolve issues in)",
            "improve": "improve (enhance the quality/performance of)",
            "update": "update (modify to incorporate new requirements for)",
            "change": "change (alter the implementation of)",
            "modify": "modify (adjust the behavior of)",
        }

        enhanced = prompt
        for old, new in replacements.items():
            # Only replace if it's a standalone word
            pattern = r"\b" + old + r"\b"
            if re.search(pattern, enhanced, re.IGNORECASE):
                enhanced = re.sub(pattern, new, enhanced, count=1, flags=re.IGNORECASE)
                break  # Only clarify one verb to avoid over-modification

        return enhanced

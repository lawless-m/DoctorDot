from guardrails.base import Guardrail
from typing import List, Dict, Optional
from pathlib import Path
import re


class HRPoliciesGuardrail(Guardrail):
    """
    Guardrail for HR policy chatbot.

    Ensures responses stay within HR policy domain.
    """

    def __init__(self, config_path: Optional[Path] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "hr_policies.yaml"
        super().__init__(config_path)

    def get_system_prompt(self) -> str:
        """Return the system prompt for Claude."""
        return self.config['system_prompt']

    def should_reject_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Determine if query is outside HR policy scope.

        Uses keyword matching as a simple heuristic.
        For production, could use embeddings or Claude itself.
        """
        query_lower = query.lower()

        # Check if strict mode is enabled
        if not self.config.get('strict_mode', True):
            return False, None

        # Check for HR-related keywords
        allowed_topics = self.config.get('allowed_topics', [])

        # If query contains any allowed topic keyword, allow it
        for topic in allowed_topics:
            if topic.lower() in query_lower:
                return False, None

        # Common non-HR patterns to explicitly reject
        rejection_patterns = [
            r'\b(weather|stock|news|sports|game)\b',
            r'\b(recipe|cook|food)\b',
            r'\b(movie|film|tv show)\b',
            r'\b(code|programming|debug)\b',
        ]

        for pattern in rejection_patterns:
            if re.search(pattern, query_lower):
                return True, "Query appears to be outside HR policy domain"

        # If no keywords match and query is longer (suggesting detailed question),
        # allow it through (retrieval will determine relevance)
        if len(query.split()) > 5:
            return False, None

        # Short queries with no HR keywords → likely off-topic
        return True, "Query doesn't appear to relate to HR policies"

    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks for Claude prompt.

        Args:
            retrieved_chunks: List of chunks from vector search

        Returns:
            Formatted XML context
        """
        if not retrieved_chunks:
            return "<hr_policies>\nNo relevant policy documents found.\n</hr_policies>"

        context_parts = ["<hr_policies>"]

        for i, chunk in enumerate(retrieved_chunks, 1):
            context_parts.append(f"\n<policy_excerpt id='{i}'>")
            context_parts.append(f"<source>{chunk['document_name']}</source>")
            if chunk.get('page_number'):
                context_parts.append(f"<page>{chunk['page_number']}</page>")
            context_parts.append(f"<content>\n{chunk['chunk_text']}\n</content>")
            context_parts.append(f"<relevance_score>{chunk['similarity']:.3f}</relevance_score>")
            context_parts.append("</policy_excerpt>")

        context_parts.append("\n</hr_policies>")

        return "\n".join(context_parts)

    def get_rejection_message(self, reason: Optional[str] = None) -> str:
        """Return user-friendly rejection message."""
        base_message = self.config['rejection_message']

        if reason:
            return f"{base_message}\n\nReason: {reason}"

        return base_message


# Factory function
def create_guardrail(name: str) -> Guardrail:
    """
    Create a guardrail instance by name.

    Args:
        name: Guardrail name (e.g., 'hr_policies', 'engineering')

    Returns:
        Guardrail instance
    """
    if name == "hr_policies":
        return HRPoliciesGuardrail()
    else:
        raise ValueError(f"Unknown guardrail: {name}")

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import yaml
from pathlib import Path


class Guardrail(ABC):
    """
    Abstract base class for domain-specific guardrails.

    Each guardrail implementation controls:
    - System prompts for Claude
    - Topic validation
    - Response formatting
    - Citation requirements
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config = self._load_config(config_path) if config_path else {}
        self.domain_name = self.config.get('domain', 'Unknown')

    def _load_config(self, config_path: Path) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Return the system prompt for Claude API.

        This defines Claude's behavior and constraints.
        """
        pass

    @abstractmethod
    def should_reject_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Determine if a query is outside the allowed scope.

        Args:
            query: User's question

        Returns:
            Tuple of (should_reject: bool, rejection_reason: Optional[str])
        """
        pass

    @abstractmethod
    def format_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Format retrieved chunks for inclusion in Claude prompt.

        Args:
            retrieved_chunks: List of chunks from vector search

        Returns:
            Formatted context string
        """
        pass

    @abstractmethod
    def get_rejection_message(self, reason: Optional[str] = None) -> str:
        """
        Return message to show user when query is rejected.

        Args:
            reason: Optional specific reason for rejection

        Returns:
            User-friendly rejection message
        """
        pass

    def get_retrieval_config(self) -> Dict:
        """
        Return configuration for retrieval parameters.

        Can be overridden to customize per-domain retrieval.
        """
        return {
            'top_k': self.config.get('top_k', 5),
            'similarity_threshold': self.config.get('similarity_threshold', 0.7)
        }

    def requires_citations(self) -> bool:
        """Whether this guardrail requires source citations."""
        return self.config.get('require_citations', True)

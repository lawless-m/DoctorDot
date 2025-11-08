from anthropic import Anthropic
from typing import List, Dict, Optional
from backend.config import settings
import logging

logger = logging.getLogger(__name__)


class ClaudeClient:
    """
    Wrapper for Anthropic Claude API.
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or settings.ANTHROPIC_API_KEY
        self.client = Anthropic(api_key=self.api_key)
        self.model = settings.CLAUDE_MODEL
        self.max_tokens = settings.CLAUDE_MAX_TOKENS
        self.temperature = settings.CLAUDE_TEMPERATURE

    def generate_response(
        self,
        system_prompt: str,
        context: str,
        user_question: str,
        conversation_history: List[Dict] = None
    ) -> tuple[str, Dict]:
        """
        Generate a response from Claude.

        Args:
            system_prompt: System instructions for Claude
            context: Retrieved document context
            user_question: User's question
            conversation_history: Previous messages (optional)

        Returns:
            Tuple of (response_text, metadata)
        """
        # Build messages array
        messages = []

        # Add conversation history if provided
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })

        # Add current query with context
        user_message = f"""{context}

<user_question>
{user_question}
</user_question>

Please answer the user's question based on the policy documents provided above.
Remember to cite your sources."""

        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system=system_prompt,
                messages=messages
            )

            # Extract response text
            response_text = response.content[0].text

            # Build metadata
            metadata = {
                "model": response.model,
                "stop_reason": response.stop_reason,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens
            }

            logger.info(
                f"Claude response generated. "
                f"Tokens: {metadata['input_tokens']} in, {metadata['output_tokens']} out"
            )

            return response_text, metadata

        except Exception as e:
            logger.error(f"Error calling Claude API: {e}")
            raise

    def check_api_key(self) -> bool:
        """Verify API key is valid."""
        try:
            # Make a minimal API call to test
            self.client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            return True
        except Exception as e:
            logger.error(f"API key validation failed: {e}")
            return False

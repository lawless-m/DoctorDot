from sentence_transformers import SentenceTransformer
import torch
from typing import List, Optional
import logging
from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    GPU-accelerated embedding generation service.
    Uses sentence-transformers with CUDA support.
    """

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        # Verify CUDA availability
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            self.device = "cpu"

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        # Load model
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"Model loaded. Embedding dimension: {self.dimension}")

    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text string

        Returns:
            List of floats representing the embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,  # Normalize for cosine similarity
            show_progress_bar=False
        )
        return embedding.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.

        Args:
            texts: List of input text strings
            batch_size: Batch size for processing (default from settings)

        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True
        )

        return embeddings.tolist()

    def get_model_info(self) -> dict:
        """Return information about the loaded model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length
        }


# Global instance (singleton pattern)
_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Get or create the global embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # API Keys
    ANTHROPIC_API_KEY: str

    # Model Configuration
    EMBEDDING_MODEL: str = "all-mpnet-base-v2"  # 768 dimensions
    EMBEDDING_DEVICE: str = "cuda"
    EMBEDDING_BATCH_SIZE: int = 32

    # Paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DUCKDB_PATH: Path = PROJECT_ROOT / "data"
    COLLECTIONS_PATH: Path = PROJECT_ROOT / "collections"
    GUARDRAILS_PATH: Path = PROJECT_ROOT / "guardrails"

    # Document Processing
    CHUNK_SIZE: int = 500  # characters per chunk
    CHUNK_OVERLAP: int = 50  # overlap between chunks

    # Retrieval Configuration
    TOP_K_RESULTS: int = 5  # number of chunks to retrieve
    SIMILARITY_THRESHOLD: float = 0.3  # minimum cosine similarity

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:5173"]  # Vite default

    # Claude Configuration
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 2000
    CLAUDE_TEMPERATURE: float = 0.0  # Deterministic for policy answers

    class Config:
        env_file = Path(__file__).parent / ".env"
        case_sensitive = True


settings = Settings()

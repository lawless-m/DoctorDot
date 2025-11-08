# HR Policy Chatbot - Detailed Implementation Plan

## Project Overview
Build a modular chatbot system that answers questions strictly based on document collections (starting with HR policies). Uses local GPU (24GB NVIDIA 3070) for embeddings, DuckDB for vector storage, and Claude API for responses.

**Key Design Principles:**
- Modular guardrails for easy domain switching
- Collection-based document organization
- GPU-accelerated local embeddings (no API costs)
- Semantic search with vector similarity
- Strict source-based responses only

---

## Technology Stack

### Backend
- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Embeddings**: sentence-transformers with CUDA support
- **Vector DB**: DuckDB with VSS extension
- **LLM**: Anthropic Claude API (claude-sonnet-4-20250514)
- **PDF Processing**: pymupdf (PyMuPDF)
- **Async**: asyncio with httpx for API calls

### Frontend
- **Framework**: React 18+ with TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **HTTP Client**: axios

### Infrastructure
- **GPU**: NVIDIA 3070 24GB (local)
- **OS**: Linux (assumed, for GPU drivers)
- **Python Version**: 3.10+
- **Node Version**: 18+

---

## Project Directory Structure

```
hr-policy-chatbot/
├── backend/
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Configuration and environment variables
│   ├── models.py               # Pydantic models for request/response
│   ├── embeddings.py           # GPU embedding service wrapper
│   ├── retrieval.py            # DuckDB vector search operations
│   ├── claude_client.py        # Anthropic API client wrapper
│   ├── document_processor.py  # PDF parsing and chunking logic
│   ├── ingest.py              # Document ingestion CLI script
│   ├── requirements.txt        # Python dependencies
│   └── .env.example           # Environment variable template
│
├── guardrails/
│   ├── __init__.py            # Package initialization
│   ├── base.py                # Abstract Guardrail base class
│   ├── hr_policies.py         # HR-specific guardrail implementation
│   ├── hr_policies.yaml       # HR guardrail configuration
│   ├── engineering.py         # Example: engineering docs guardrail
│   ├── engineering.yaml       # Example: engineering config
│   └── README.md              # Documentation for creating new guardrails
│
├── collections/
│   ├── hr_policies/           # HR policy PDF documents
│   │   ├── .gitkeep
│   │   └── (PDFs placed here by user)
│   ├── engineering_docs/      # Placeholder for future use
│   │   └── .gitkeep
│   └── legal_contracts/       # Placeholder for future use
│       └── .gitkeep
│
├── data/
│   ├── .gitkeep
│   └── (generated .duckdb files stored here)
│       # e.g., hr_policies.duckdb, engineering_docs.duckdb
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx    # Main chat UI component
│   │   │   ├── MessageList.tsx      # Display message history
│   │   │   ├── MessageInput.tsx     # User input field
│   │   │   ├── MessageBubble.tsx    # Individual message display
│   │   │   └── CitationDisplay.tsx  # Show source citations
│   │   ├── services/
│   │   │   └── api.ts               # Backend API client
│   │   ├── types/
│   │   │   └── index.ts             # TypeScript type definitions
│   │   ├── hooks/
│   │   │   └── useChat.ts           # Custom hook for chat logic
│   │   ├── App.tsx                  # Root component
│   │   ├── main.tsx                 # Entry point
│   │   └── index.css                # Global styles
│   ├── public/
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── scripts/
│   ├── setup.sh               # Initial project setup
│   ├── run_ingest.sh          # Wrapper for document ingestion
│   └── check_gpu.py           # Verify GPU availability
│
├── tests/
│   ├── test_embeddings.py     # Test embedding generation
│   ├── test_retrieval.py      # Test vector search
│   ├── test_guardrails.py     # Test guardrail logic
│   ├── test_document_processor.py
│   └── test_api.py            # API endpoint tests
│
├── docs/
│   ├── SETUP.md               # Setup instructions
│   ├── USAGE.md               # User guide
│   └── ADDING_COLLECTIONS.md  # Guide for new document collections
│
├── README.md
├── .gitignore
├── .env.example
└── docker-compose.yml         # Optional: containerized deployment
```

---

## Phase 1: Backend Core Infrastructure

### 1.1 Configuration Management (`backend/config.py`)

**Purpose**: Centralize all configuration and environment variables

**Implementation**:
```python
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
    SIMILARITY_THRESHOLD: float = 0.7  # minimum cosine similarity
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: list = ["http://localhost:5173"]  # Vite default
    
    # Claude Configuration
    CLAUDE_MODEL: str = "claude-sonnet-4-20250514"
    CLAUDE_MAX_TOKENS: int = 2000
    CLAUDE_TEMPERATURE: float = 0.0  # Deterministic for policy answers
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

**Environment Variables** (`.env.example`):
```bash
ANTHROPIC_API_KEY=sk-ant-...
EMBEDDING_MODEL=all-mpnet-base-v2
EMBEDDING_DEVICE=cuda
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K_RESULTS=5
```

---

### 1.2 Data Models (`backend/models.py`)

**Purpose**: Define all request/response schemas using Pydantic

**Models to Implement**:

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Chat Models
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_text: str
    relevance_score: float
    chunk_id: str

class QueryRequest(BaseModel):
    question: str
    collection_name: str = "hr_policies"
    conversation_history: List[Message] = []
    include_citations: bool = True

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    guardrail_triggered: bool = False
    rejection_reason: Optional[str] = None
    processing_time_ms: float

# Ingestion Models
class IngestRequest(BaseModel):
    collection_name: str
    guardrail_name: str
    force_reindex: bool = False

class DocumentStatus(BaseModel):
    filename: str
    status: str  # "success", "error", "skipped"
    chunks_created: int
    error_message: Optional[str] = None

class IngestResponse(BaseModel):
    status: str
    collection_name: str
    documents_processed: int
    total_chunks: int
    document_details: List[DocumentStatus]
    processing_time_seconds: float
    errors: List[str] = []

# Health Check
class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    cuda_version: Optional[str] = None
    embedding_model_loaded: bool
    active_collection: Optional[str] = None
    duckdb_status: str

# Embedding Models
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model_name: str
    dimension: int
```

---

### 1.3 Embedding Service (`backend/embeddings.py`)

**Purpose**: Manage GPU-based embedding generation using sentence-transformers

**Key Features**:
- Load model once on GPU at startup
- Batch processing for efficiency
- Error handling for OOM scenarios
- Model caching

**Implementation**:
```python
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Union
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
```

**GPU Memory Management**:
- Model loads ~420MB for all-mpnet-base-v2
- Batch size of 32 uses ~2-3GB VRAM
- Plenty of headroom on 24GB GPU

**Alternative Models** (for future consideration):
- `all-MiniLM-L6-v2`: Faster, smaller (384 dim), good quality
- `all-mpnet-base-v2`: Balanced (768 dim) ← **RECOMMENDED**
- `bge-large-en-v1.5`: Best quality (1024 dim), slower

---

### 1.4 Document Processing (`backend/document_processor.py`)

**Purpose**: Extract text from PDFs and chunk intelligently

**Key Features**:
- PDF text extraction with metadata
- Smart chunking (respects sentence boundaries)
- Overlap between chunks for context
- Metadata preservation (filename, page numbers)

**Implementation**:
```python
import pymupdf  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
import re
from backend.config import settings
import logging

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Represents a chunk of text from a document."""
    def __init__(
        self,
        text: str,
        document_name: str,
        page_number: Optional[int] = None,
        chunk_index: int = 0,
        metadata: Optional[Dict] = None
    ):
        self.text = text
        self.document_name = document_name
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.metadata = metadata or {}
        self.chunk_id = f"{document_name}_p{page_number}_c{chunk_index}"

class DocumentProcessor:
    """
    Processes PDF documents into searchable chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extract text from PDF with page-level granularity.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dicts with 'page_number' and 'text'
        """
        try:
            doc = pymupdf.open(pdf_path)
            pages = []
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text()
                # Clean up text
                text = self._clean_text(text)
                
                if text.strip():  # Only add non-empty pages
                    pages.append({
                        'page_number': page_num,
                        'text': text
                    })
            
            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
            return pages
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove page numbers (common pattern)
        text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
        return text.strip()
    
    def chunk_text(
        self,
        text: str,
        document_name: str,
        page_number: Optional[int] = None
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks.
        
        Uses sentence-aware chunking to avoid breaking mid-sentence.
        
        Args:
            text: Text to chunk
            document_name: Name of source document
            page_number: Page number (if applicable)
            
        Returns:
            List of DocumentChunk objects
        """
        # Split into sentences
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    document_name=document_name,
                    page_number=page_number,
                    chunk_index=chunk_index
                ))
                
                chunk_index += 1
                
                # Create overlap by keeping last few sentences
                overlap_sentences = self._get_overlap_sentences(
                    current_chunk,
                    self.chunk_overlap
                )
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in current_chunk)
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(DocumentChunk(
                text=chunk_text,
                document_name=document_name,
                page_number=page_number,
                chunk_index=chunk_index
            ))
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting (can be improved with nltk/spacy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_overlap_sentences(
        self,
        sentences: List[str],
        overlap_chars: int
    ) -> List[str]:
        """Get the last N sentences that fit within overlap_chars."""
        overlap_sentences = []
        total_length = 0
        
        for sentence in reversed(sentences):
            if total_length + len(sentence) > overlap_chars:
                break
            overlap_sentences.insert(0, sentence)
            total_length += len(sentence)
        
        return overlap_sentences
    
    def process_document(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Process entire PDF document into chunks.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of all chunks from the document
        """
        document_name = pdf_path.stem
        pages = self.extract_text_from_pdf(pdf_path)
        
        all_chunks = []
        for page_data in pages:
            page_chunks = self.chunk_text(
                text=page_data['text'],
                document_name=document_name,
                page_number=page_data['page_number']
            )
            all_chunks.extend(page_chunks)
        
        logger.info(
            f"Processed {document_name}: "
            f"{len(pages)} pages → {len(all_chunks)} chunks"
        )
        
        return all_chunks
    
    def process_collection(self, collection_path: Path) -> Dict[str, List[DocumentChunk]]:
        """
        Process all PDFs in a collection directory.
        
        Args:
            collection_path: Path to directory containing PDFs
            
        Returns:
            Dict mapping filename to list of chunks
        """
        pdf_files = list(collection_path.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {collection_path}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files in {collection_path}")
        
        results = {}
        for pdf_path in pdf_files:
            try:
                chunks = self.process_document(pdf_path)
                results[pdf_path.name] = chunks
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                results[pdf_path.name] = []
        
        total_chunks = sum(len(chunks) for chunks in results.values())
        logger.info(f"Total chunks created: {total_chunks}")
        
        return results
```

**Chunking Strategy**:
- Default: 500 characters per chunk with 50 character overlap
- Respects sentence boundaries
- Preserves context between chunks
- Metadata tracking for citations

---

### 1.5 Vector Storage with DuckDB (`backend/retrieval.py`)

**Purpose**: Store and search embeddings using DuckDB VSS extension

**Key Features**:
- Create vector database per collection
- Insert document chunks with embeddings
- Cosine similarity search
- Metadata filtering

**Implementation**:
```python
import duckdb
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from backend.config import settings
from backend.document_processor import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    """
    DuckDB-based vector store with VSS extension.
    """
    
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.db_path = settings.DUCKDB_PATH / f"{collection_name}.duckdb"
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
    
    def connect(self):
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")
        
        # Install and load VSS extension
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")
    
    def initialize_schema(self, embedding_dimension: int):
        """
        Create the schema for storing document chunks and embeddings.
        
        Args:
            embedding_dimension: Dimensionality of embedding vectors
        """
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                chunk_id VARCHAR PRIMARY KEY,
                document_name VARCHAR NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding FLOAT[{dim}] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """.format(dim=embedding_dimension))
        
        # Create HNSW index for fast similarity search
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS embedding_idx 
            ON documents USING HNSW (embedding)
            WITH (metric = 'cosine');
        """)
        
        logger.info(f"Initialized schema with embedding dimension {embedding_dimension}")
    
    def insert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]]
    ):
        """
        Insert document chunks with their embeddings.
        
        Args:
            chunks: List of DocumentChunk objects
            embeddings: Corresponding embedding vectors
        """
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings")
        
        data = [
            (
                chunk.chunk_id,
                chunk.document_name,
                chunk.page_number,
                chunk.chunk_index,
                chunk.text,
                embedding
            )
            for chunk, embedding in zip(chunks, embeddings)
        ]
        
        self.conn.executemany("""
            INSERT INTO documents 
            (chunk_id, document_name, page_number, chunk_index, chunk_text, embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data)
        
        logger.info(f"Inserted {len(chunks)} chunks into database")
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = None,
        similarity_threshold: float = None
    ) -> List[Dict]:
        """
        Search for similar chunks using cosine similarity.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of dicts containing chunk data and similarity scores
        """
        top_k = top_k or settings.TOP_K_RESULTS
        similarity_threshold = similarity_threshold or settings.SIMILARITY_THRESHOLD
        
        # DuckDB VSS search with cosine similarity
        results = self.conn.execute("""
            SELECT 
                chunk_id,
                document_name,
                page_number,
                chunk_index,
                chunk_text,
                array_cosine_similarity(embedding, ?::FLOAT[]) as similarity
            FROM documents
            WHERE array_cosine_similarity(embedding, ?::FLOAT[]) >= ?
            ORDER BY similarity DESC
            LIMIT ?
        """, [query_embedding, query_embedding, similarity_threshold, top_k]).fetchall()
        
        # Convert to list of dicts
        results_list = [
            {
                'chunk_id': row[0],
                'document_name': row[1],
                'page_number': row[2],
                'chunk_index': row[3],
                'chunk_text': row[4],
                'similarity': float(row[5])
            }
            for row in results
        ]
        
        logger.info(f"Found {len(results_list)} results above threshold {similarity_threshold}")
        return results_list
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store."""
        result = self.conn.execute("""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT document_name) as total_documents
            FROM documents
        """).fetchone()
        
        return {
            'total_chunks': result[0],
            'total_documents': result[1],
            'collection_name': self.collection_name,
            'db_path': str(self.db_path)
        }
    
    def clear(self):
        """Delete all data from the collection."""
        self.conn.execute("DELETE FROM documents;")
        logger.warning(f"Cleared all data from {self.collection_name}")
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Closed database connection")

def create_vector_store(collection_name: str) -> VectorStore:
    """
    Factory function to create and connect a VectorStore.
    
    Args:
        collection_name: Name of the document collection
        
    Returns:
        Connected VectorStore instance
    """
    store = VectorStore(collection_name)
    store.connect()
    return store
```

**DuckDB VSS Features**:
- HNSW index for fast approximate nearest neighbor search
- Cosine similarity built-in
- SQL interface for complex queries
- Persistent storage in single file per collection

---

## Phase 2: Guardrails System

### 2.1 Base Guardrail Class (`guardrails/base.py`)

**Purpose**: Abstract interface for domain-specific guardrails

**Implementation**:
```python
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
```

---

### 2.2 HR Policies Guardrail (`guardrails/hr_policies.yaml`)

**Purpose**: Configuration for HR policy domain

```yaml
domain: "HR Policies"
description: "Company HR policies, benefits, procedures, and employee guidelines"

# Retrieval parameters
top_k: 5
similarity_threshold: 0.7

# Behavior flags
require_citations: true
allow_conversational_queries: true
strict_mode: true  # Reject any non-HR queries

# Allowed topic keywords (for basic filtering)
allowed_topics:
  - "benefits"
  - "leave"
  - "vacation"
  - "sick"
  - "holiday"
  - "remote"
  - "work from home"
  - "hours"
  - "pay"
  - "salary"
  - "compensation"
  - "performance"
  - "review"
  - "disciplinary"
  - "grievance"
  - "harassment"
  - "discrimination"
  - "resignation"
  - "termination"
  - "probation"
  - "training"
  - "development"
  - "dress code"
  - "expenses"
  - "travel"
  - "equipment"
  - "policy"
  - "procedure"

# Rejection message template
rejection_message: |
  I can only answer questions about our company's HR policies and procedures.
  
  Your question appears to be about something outside of HR policies. 
  
  I can help with topics like:
  - Employee benefits and leave policies
  - Working hours and remote work
  - Performance reviews and development
  - Expenses and equipment
  - Company procedures and guidelines
  
  Please rephrase your question to relate to HR policies, or contact the HR team directly.

# System prompt template
system_prompt: |
  You are an HR Policy Assistant for the company. Your role is to answer employee questions 
  based STRICTLY on the official HR policy documents provided.
  
  CRITICAL RULES:
  1. ONLY answer based on the provided policy documents
  2. If the answer is not in the documents, say "I don't have information about that in our HR policies"
  3. NEVER make up or infer policies that aren't explicitly stated
  4. ALWAYS cite the specific policy document and section
  5. If a policy is ambiguous, acknowledge the ambiguity
  6. For sensitive topics (discipline, termination), be especially careful to quote policy exactly
  7. Remain professional and neutral in tone
  
  SCOPE LIMITATIONS:
  - Only discuss company HR policies and procedures
  - Do not provide legal advice
  - Do not interpret laws or regulations beyond what's in the policies
  - Do not discuss specific individuals or cases
  - Refer complex or sensitive situations to the HR team
  
  RESPONSE FORMAT:
  - Be clear and concise
  - Use bullet points for multi-part answers
  - Always include the source policy name
  - If referencing multiple policies, cite each one
```

---

### 2.3 HR Policies Guardrail Implementation (`guardrails/hr_policies.py`)

```python
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
```

---

### 2.4 Example: Engineering Docs Guardrail (`guardrails/engineering.yaml`)

**Purpose**: Show how to create additional guardrails

```yaml
domain: "Engineering Documentation"
description: "Technical documentation, API references, and engineering guidelines"

top_k: 7
similarity_threshold: 0.65  # Lower threshold for technical content

require_citations: true
allow_conversational_queries: true
strict_mode: false  # Allow more flexibility

allowed_topics:
  - "api"
  - "endpoint"
  - "database"
  - "architecture"
  - "deployment"
  - "testing"
  - "framework"
  - "library"
  - "configuration"
  - "documentation"

rejection_message: |
  I can only answer questions about our engineering documentation and technical systems.
  
  Please ask about our APIs, architecture, deployment procedures, or technical guidelines.

system_prompt: |
  You are a Technical Documentation Assistant. Answer questions based on the engineering 
  documentation provided.
  
  RULES:
  1. Only answer based on provided documentation
  2. Include code examples when relevant
  3. Cite specific documentation sections
  4. If information is outdated or missing, acknowledge it
  5. For complex technical questions, provide step-by-step explanations
```

---

## Phase 3: Claude API Integration

### 3.1 Claude Client (`backend/claude_client.py`)

**Purpose**: Wrapper for Anthropic API calls

**Implementation**:
```python
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
```

---

## Phase 4: Document Ingestion

### 4.1 Ingestion Script (`backend/ingest.py`)

**Purpose**: CLI tool to process documents and populate vector store

**Implementation**:
```python
import argparse
import sys
from pathlib import Path
import time
import logging
from typing import List

from backend.config import settings
from backend.document_processor import DocumentProcessor, DocumentChunk
from backend.embeddings import get_embedding_service
from backend.retrieval import create_vector_store
from backend.models import DocumentStatus, IngestResponse
from guardrails.hr_policies import create_guardrail

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """
    End-to-end pipeline for ingesting documents into vector store.
    """
    
    def __init__(
        self,
        collection_name: str,
        guardrail_name: str,
        force_reindex: bool = False
    ):
        self.collection_name = collection_name
        self.guardrail_name = guardrail_name
        self.force_reindex = force_reindex
        
        # Initialize components
        logger.info("Initializing ingestion pipeline...")
        self.embedding_service = get_embedding_service()
        self.document_processor = DocumentProcessor()
        self.vector_store = create_vector_store(collection_name)
        self.guardrail = create_guardrail(guardrail_name)
        
        # Paths
        self.collection_path = settings.COLLECTIONS_PATH / collection_name
        
        if not self.collection_path.exists():
            raise ValueError(f"Collection path does not exist: {self.collection_path}")
    
    def run(self) -> IngestResponse:
        """
        Execute the full ingestion pipeline.
        
        Returns:
            IngestResponse with results
        """
        start_time = time.time()
        
        logger.info(f"Starting ingestion for collection: {self.collection_name}")
        logger.info(f"Using guardrail: {self.guardrail_name}")
        
        # Check if we should clear existing data
        if self.force_reindex:
            logger.warning("Force reindex enabled - clearing existing data")
            self.vector_store.clear()
        
        # Initialize database schema
        embedding_dim = self.embedding_service.dimension
        self.vector_store.initialize_schema(embedding_dim)
        
        # Process all documents
        document_results = self.document_processor.process_collection(self.collection_path)
        
        # Track results
        document_statuses: List[DocumentStatus] = []
        total_chunks = 0
        errors = []
        
        # Process each document
        for filename, chunks in document_results.items():
            try:
                if not chunks:
                    document_statuses.append(DocumentStatus(
                        filename=filename,
                        status="skipped",
                        chunks_created=0,
                        error_message="No content extracted"
                    ))
                    continue
                
                # Generate embeddings
                logger.info(f"Generating embeddings for {filename} ({len(chunks)} chunks)...")
                chunk_texts = [chunk.text for chunk in chunks]
                embeddings = self.embedding_service.embed_batch(chunk_texts)
                
                # Insert into vector store
                logger.info(f"Inserting {len(chunks)} chunks into vector store...")
                self.vector_store.insert_chunks(chunks, embeddings)
                
                document_statuses.append(DocumentStatus(
                    filename=filename,
                    status="success",
                    chunks_created=len(chunks)
                ))
                
                total_chunks += len(chunks)
                
            except Exception as e:
                error_msg = f"Error processing {filename}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                
                document_statuses.append(DocumentStatus(
                    filename=filename,
                    status="error",
                    chunks_created=0,
                    error_message=str(e)
                ))
        
        # Close database connection
        self.vector_store.close()
        
        processing_time = time.time() - start_time
        
        # Build response
        response = IngestResponse(
            status="completed" if not errors else "completed_with_errors",
            collection_name=self.collection_name,
            documents_processed=len(document_results),
            total_chunks=total_chunks,
            document_details=document_statuses,
            processing_time_seconds=round(processing_time, 2),
            errors=errors
        )
        
        logger.info(f"Ingestion completed in {processing_time:.2f} seconds")
        logger.info(f"Total documents: {response.documents_processed}")
        logger.info(f"Total chunks: {response.total_chunks}")
        
        if errors:
            logger.warning(f"Completed with {len(errors)} errors")
        
        return response

def main():
    parser = argparse.ArgumentParser(
        description="Ingest documents into vector store"
    )
    parser.add_argument(
        "--collection",
        required=True,
        help="Name of document collection (e.g., hr_policies)"
    )
    parser.add_argument(
        "--guardrail",
        required=True,
        help="Name of guardrail to use (e.g., hr_policies)"
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Clear existing data and reindex from scratch"
    )
    
    args = parser.parse_args()
    
    try:
        pipeline = DocumentIngestionPipeline(
            collection_name=args.collection,
            guardrail_name=args.guardrail,
            force_reindex=args.force_reindex
        )
        
        result = pipeline.run()
        
        # Print summary
        print("\n" + "="*60)
        print("INGESTION SUMMARY")
        print("="*60)
        print(f"Collection: {result.collection_name}")
        print(f"Status: {result.status}")
        print(f"Documents Processed: {result.documents_processed}")
        print(f"Total Chunks: {result.total_chunks}")
        print(f"Processing Time: {result.processing_time_seconds}s")
        
        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for error in result.errors:
                print(f"  - {error}")
        
        print("\nDocument Details:")
        for doc in result.document_details:
            status_emoji = "✓" if doc.status == "success" else "✗"
            print(f"  {status_emoji} {doc.filename}: {doc.chunks_created} chunks ({doc.status})")
            if doc.error_message:
                print(f"      Error: {doc.error_message}")
        
        print("="*60 + "\n")
        
        sys.exit(0 if not result.errors else 1)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Ingest HR policies
python backend/ingest.py --collection hr_policies --guardrail hr_policies

# Force reindex
python backend/ingest.py --collection hr_policies --guardrail hr_policies --force-reindex
```

---

## Phase 5: FastAPI Application

### 5.1 Main API (`backend/main.py`)

**Purpose**: FastAPI application with all endpoints

**Implementation**:
```python
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import time
import logging
from contextlib import asynccontextmanager

from backend.config import settings
from backend.models import (
    QueryRequest, QueryResponse, Citation,
    HealthResponse, IngestRequest, IngestResponse
)
from backend.embeddings import get_embedding_service
from backend.retrieval import create_vector_store
from backend.claude_client import ClaudeClient
from guardrails.hr_policies import create_guardrail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
app_state = {
    "embedding_service": None,
    "vector_store": None,
    "claude_client": None,
    "guardrail": None,
    "current_collection": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    """
    # Startup
    logger.info("Starting up application...")
    
    try:
        # Initialize embedding service (loads model on GPU)
        logger.info("Loading embedding model on GPU...")
        app_state["embedding_service"] = get_embedding_service()
        
        # Initialize Claude client
        logger.info("Initializing Claude client...")
        app_state["claude_client"] = ClaudeClient()
        
        # Load default collection and guardrail
        default_collection = "hr_policies"
        default_guardrail = "hr_policies"
        
        logger.info(f"Loading collection: {default_collection}")
        app_state["vector_store"] = create_vector_store(default_collection)
        app_state["current_collection"] = default_collection
        
        logger.info(f"Loading guardrail: {default_guardrail}")
        app_state["guardrail"] = create_guardrail(default_guardrail)
        
        logger.info("Application startup complete!")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    if app_state["vector_store"]:
        app_state["vector_store"].close()

app = FastAPI(
    title="HR Policy Chatbot API",
    description="Query company policies using semantic search and Claude",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    """
    import torch
    
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        cuda_version=torch.version.cuda if torch.cuda.is_available() else None,
        embedding_model_loaded=app_state["embedding_service"] is not None,
        active_collection=app_state["current_collection"],
        duckdb_status="connected" if app_state["vector_store"] else "disconnected"
    )

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the chatbot with a question.
    
    Process:
    1. Check guardrails
    2. Embed question
    3. Search vector store
    4. Call Claude API
    5. Return response with citations
    """
    start_time = time.time()
    
    try:
        # Get components
        embedding_service = app_state["embedding_service"]
        vector_store = app_state["vector_store"]
        claude_client = app_state["claude_client"]
        guardrail = app_state["guardrail"]
        
        # Check if query should be rejected
        should_reject, rejection_reason = guardrail.should_reject_query(request.question)
        
        if should_reject:
            logger.info(f"Query rejected: {rejection_reason}")
            return QueryResponse(
                answer=guardrail.get_rejection_message(rejection_reason),
                citations=[],
                guardrail_triggered=True,
                rejection_reason=rejection_reason,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        # Embed the question
        logger.info(f"Embedding question: {request.question[:100]}...")
        query_embedding = embedding_service.embed_text(request.question)
        
        # Search vector store
        logger.info("Searching vector store...")
        retrieval_config = guardrail.get_retrieval_config()
        search_results = vector_store.search(
            query_embedding=query_embedding,
            top_k=retrieval_config['top_k'],
            similarity_threshold=retrieval_config['similarity_threshold']
        )
        
        if not search_results:
            logger.warning("No relevant documents found")
            return QueryResponse(
                answer="I couldn't find any relevant information in our HR policies to answer your question. Please try rephrasing, or contact the HR team directly.",
                citations=[],
                guardrail_triggered=False,
                processing_time_ms=round((time.time() - start_time) * 1000, 2)
            )
        
        logger.info(f"Found {len(search_results)} relevant chunks")
        
        # Format context for Claude
        context = guardrail.format_context(search_results)
        
        # Get system prompt
        system_prompt = guardrail.get_system_prompt()
        
        # Call Claude API
        logger.info("Calling Claude API...")
        answer, claude_metadata = claude_client.generate_response(
            system_prompt=system_prompt,
            context=context,
            user_question=request.question,
            conversation_history=request.conversation_history
        )
        
        # Build citations
        citations = [
            Citation(
                document_name=chunk['document_name'],
                page_number=chunk.get('page_number'),
                chunk_text=chunk['chunk_text'][:200] + "...",  # Truncate for display
                relevance_score=chunk['similarity'],
                chunk_id=chunk['chunk_id']
            )
            for chunk in search_results
        ]
        
        processing_time = round((time.time() - start_time) * 1000, 2)
        
        logger.info(f"Query processed successfully in {processing_time}ms")
        
        return QueryResponse(
            answer=answer,
            citations=citations if request.include_citations else [],
            guardrail_triggered=False,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """
    Get statistics about the current collection.
    """
    try:
        vector_store = app_state["vector_store"]
        if not vector_store:
            raise HTTPException(status_code=503, detail="Vector store not initialized")
        
        stats = vector_store.get_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """
    List available document collections.
    """
    collections_path = settings.COLLECTIONS_PATH
    collections = [
        d.name for d in collections_path.iterdir() 
        if d.is_dir() and not d.name.startswith('.')
    ]
    return {"collections": collections}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
```

---

## Phase 6: React Frontend

### 6.1 TypeScript Types (`frontend/src/types/index.ts`)

```typescript
export interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  citations?: Citation[];
}

export interface Citation {
  document_name: string;
  page_number?: number;
  chunk_text: string;
  relevance_score: number;
  chunk_id: string;
}

export interface QueryResponse {
  answer: string;
  citations: Citation[];
  guardrail_triggered: boolean;
  rejection_reason?: string;
  processing_time_ms: number;
}

export interface HealthStatus {
  status: string;
  gpu_available: boolean;
  cuda_version?: string;
  embedding_model_loaded: boolean;
  active_collection?: string;
  duckdb_status: string;
}
```

---

### 6.2 API Client (`frontend/src/services/api.ts`)

```typescript
import axios from 'axios';
import { QueryResponse, HealthStatus } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const chatAPI = {
  async query(
    question: string,
    collection: string = 'hr_policies',
    conversationHistory: Array<{role: string; content: string}> = []
  ): Promise<QueryResponse> {
    const response = await api.post<QueryResponse>('/query', {
      question,
      collection_name: collection,
      conversation_history: conversationHistory,
      include_citations: true,
    });
    return response.data;
  },

  async getHealth(): Promise<HealthStatus> {
    const response = await api.get<HealthStatus>('/health');
    return response.data;
  },

  async getStats(): Promise<any> {
    const response = await api.get('/stats');
    return response.data;
  },

  async listCollections(): Promise<string[]> {
    const response = await api.get<{collections: string[]}>('/collections');
    return response.data.collections;
  },
};
```

---

### 6.3 Chat Interface Component (`frontend/src/components/ChatInterface.tsx`)

```typescript
import React, { useState, useRef, useEffect } from 'react';
import { Message } from '../types';
import { chatAPI } from '../services/api';
import MessageList from './MessageList';
import MessageInput from './MessageInput';

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (content: string) => {
    if (!content.trim()) return;

    // Add user message
    const userMessage: Message = {
      role: 'user',
      content,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const response = await chatAPI.query(
        content,
        'hr_policies',
        messages.map(m => ({ role: m.role, content: m.content }))
      );

      // Add assistant response
      const assistantMessage: Message = {
        role: 'assistant',
        content: response.answer,
        timestamp: new Date(),
        citations: response.citations,
      };

      setMessages(prev => [...prev, assistantMessage]);

    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to get response');
      console.error('Error querying chatbot:', err);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100">
      {/* Header */}
      <div className="bg-white shadow-sm p-4">
        <h1 className="text-2xl font-bold text-gray-800">HR Policy Assistant</h1>
        <p className="text-sm text-gray-600">Ask questions about company policies</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        <MessageList messages={messages} />
        <div ref={messagesEndRef} />
      </div>

      {/* Error Display */}
      {error && (
        <div className="mx-4 mb-2 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
          {error}
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t p-4">
        <MessageInput 
          onSend={handleSendMessage}
          disabled={isLoading}
        />
      </div>
    </div>
  );
};

export default ChatInterface;
```

---

### 6.4 Message Display (`frontend/src/components/MessageList.tsx`)

```typescript
import React from 'react';
import { Message } from '../types';
import MessageBubble from './MessageBubble';

interface MessageListProps {
  messages: Message[];
}

const MessageList: React.FC<MessageListProps> = ({ messages }) => {
  if (messages.length === 0) {
    return (
      <div className="text-center text-gray-500 mt-8">
        <p className="text-lg mb-2">👋 Hello! I'm your HR Policy Assistant</p>
        <p className="text-sm">Ask me anything about our company policies</p>
        <div className="mt-4 text-sm text-gray-400">
          <p>Example questions:</p>
          <ul className="mt-2 space-y-1">
            <li>"What is the remote work policy?"</li>
            <li>"How many vacation days do I get?"</li>
            <li>"What's the process for requesting leave?"</li>
          </ul>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {messages.map((message, index) => (
        <MessageBubble key={index} message={message} />
      ))}
    </div>
  );
};

export default MessageList;
```

---

### 6.5 Dependencies

**Backend (`backend/requirements.txt`)**:
```
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
anthropic==0.7.8
sentence-transformers==2.2.2
torch==2.1.0
duckdb==0.9.2
pymupdf==1.23.8
pyyaml==6.0.1
python-dotenv==1.0.0
httpx==0.25.2
```

**Frontend (`frontend/package.json`)**:
```json
{
  "name": "hr-policy-chatbot-frontend",
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc && vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "axios": "^1.6.2"
  },
  "devDependencies": {
    "@types/react": "^18.2.43",
    "@types/react-dom": "^18.2.17",
    "@vitejs/plugin-react": "^4.2.1",
    "typescript": "^5.3.3",
    "vite": "^5.0.8",
    "tailwindcss": "^3.3.6",
    "autoprefixer": "^10.4.16",
    "postcss": "^8.4.32"
  }
}
```

---

## Phase 7: Setup and Deployment

### 7.1 Setup Script (`scripts/setup.sh`)

```bash
#!/bin/bash

echo "Setting up HR Policy Chatbot..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r backend/requirements.txt

# Create necessary directories
mkdir -p data
mkdir -p collections/hr_policies
mkdir -p collections/engineering_docs
mkdir -p collections/legal_contracts

# Copy environment template
if [ ! -f backend/.env ]; then
    cp backend/.env.example backend/.env
    echo "Created .env file - please add your ANTHROPIC_API_KEY"
fi

# Install frontend dependencies
cd frontend
npm install
cd ..

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Add your ANTHROPIC_API_KEY to backend/.env"
echo "2. Place PDF files in collections/hr_policies/"
echo "3. Run: python backend/ingest.py --collection hr_policies --guardrail hr_policies"
echo "4. Start backend: python backend/main.py"
echo "5. Start frontend: cd frontend && npm run dev"
```

---

### 7.2 README (`README.md`)

```markdown
# HR Policy Chatbot

A modular chatbot system for querying document collections using semantic search and Claude AI.

## Features

- 🔍 Semantic search with GPU-accelerated embeddings
- 🛡️ Modular guardrails for domain-specific constraints
- 📚 Support for multiple document collections
- 🎯 Source citations for all responses
- ⚡ Fast vector similarity search with DuckDB
- 🤖 Powered by Claude Sonnet 4

## Quick Start

### 1. Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Configure

Add your Anthropic API key to `backend/.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### 3. Add Documents

Place PDF files in `collections/hr_policies/`

### 4. Ingest Documents

```bash
source venv/bin/activate
python backend/ingest.py --collection hr_policies --guardrail hr_policies
```

### 5. Run

Terminal 1 (Backend):
```bash
python backend/main.py
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Open http://localhost:5173

## Architecture

```
┌──────────────┐
│   Frontend   │
│   (React)    │
└──────┬───────┘
       │
       │ HTTP
       │
┌──────▼───────────────────────────────────┐
│         FastAPI Backend                  │
│  ┌────────────┐  ┌──────────────┐       │
│  │ Embeddings │  │  Guardrails  │       │
│  │  (GPU)     │  │   (Rules)    │       │
│  └────────────┘  └──────────────┘       │
│                                          │
│  ┌────────────┐  ┌──────────────┐       │
│  │  DuckDB    │  │   Claude     │       │
│  │   (VSS)    │  │     API      │       │
│  └────────────┘  └──────────────┘       │
└──────────────────────────────────────────┘
       │                    │
       │                    │
  ┌────▼─────┐         ┌────▼──────┐
  │ PDF Docs │         │ Anthropic │
  └──────────┘         └───────────┘
```

## Adding New Collections

1. Create directory: `collections/your_collection/`
2. Add PDF files
3. Create guardrail: `guardrails/your_collection.yaml`
4. Implement guardrail: `guardrails/your_collection.py`
5. Ingest: `python backend/ingest.py --collection your_collection --guardrail your_collection`

See `docs/ADDING_COLLECTIONS.md` for details.

## GPU Requirements

- CUDA-compatible GPU (tested on NVIDIA 3070 24GB)
- CUDA Toolkit 11.8+
- ~2-3GB VRAM for embeddings

## License

MIT
```

---

## Testing Strategy

### Unit Tests

```python
# tests/test_embeddings.py
def test_embedding_generation():
    service = EmbeddingService()
    text = "This is a test"
    embedding = service.embed_text(text)
    assert len(embedding) == 768  # all-mpnet-base-v2
    assert all(isinstance(x, float) for x in embedding)

# tests/test_guardrails.py
def test_hr_guardrail_rejection():
    guardrail = HRPoliciesGuardrail()
    should_reject, reason = guardrail.should_reject_query("What's the weather?")
    assert should_reject == True

# tests/test_retrieval.py
def test_vector_search():
    store = VectorStore("test_collection")
    # Insert test data
    # Search
    # Assert results
```

---

## Performance Considerations

**Embedding Generation:**
- Batch size 32: ~100ms per batch on 3070
- 200 chunks: ~2 seconds total

**Vector Search:**
- DuckDB HNSW index: <10ms for top-5
- Scales well to 10K+ chunks

**Claude API:**
- Latency: 1-3 seconds typical
- Token costs: ~$0.003 per query (estimate)

**Total Query Time:**
- Embedding: ~50ms
- Search: ~10ms
- Claude: ~2000ms
- **Total: ~2-3 seconds**

---

## Security Considerations

1. **API Key Protection**: Never commit `.env` files
2. **Input Validation**: Sanitize user queries
3. **Rate Limiting**: Add to production deployment
4. **CORS**: Configure properly for production
5. **Document Access**: Ensure proper authorization

---

## Future Enhancements

1. **Multi-turn Context**: Maintain conversation context
2. **Feedback Loop**: Let users rate responses
3. **Analytics**: Track common questions
4. **Admin Dashboard**: Manage collections via UI
5. **Authentication**: Add user auth for production
6. **Streaming Responses**: Use Claude streaming for faster UX

---

## Troubleshooting

**GPU Not Detected:**
```bash
python scripts/check_gpu.py
```

**DuckDB VSS Issues:**
```bash
pip install duckdb --upgrade
```

**Embedding Model Download:**
Models auto-download on first use. Pre-download:
```python
from sentence_transformers import SentenceTransformer
SentenceTransformer("all-mpnet-base-v2")
```

---

## Support

For issues or questions:
1. Check logs: Backend outputs to console
2. Verify GPU: `nvidia-smi`
3. Test API: `curl http://localhost:8000/health`

---

**End of Implementation Plan**
```

[View the complete implementation plan](computer:///mnt/user-data/outputs/HR_CHATBOT_IMPLEMENTATION_PLAN.md)

This is a comprehensive implementation plan that Claude (or any developer) can follow to build your HR policy chatbot. The plan includes:

1. **Complete architecture** with Python backend + React frontend
2. **Modular guardrails** system for easy domain switching
3. **Collection-based** document organization
4. **Full code examples** for all major components
5. **GPU-accelerated embeddings** with sentence-transformers
6. **DuckDB VSS** for vector storage and search
7. **Claude API integration** with proper prompt engineering
8. **Document processing** pipeline with PDF parsing and chunking
9. **FastAPI backend** with all necessary endpoints
10. **React frontend** with TypeScript
11. **Setup scripts** and documentation
12. **Testing strategy** and troubleshooting guide

The system is designed to be:
- **Modular**: Easy to swap HR policies for engineering docs or legal contracts
- **Scalable**: Works for 20 docs now, handles hundreds later
- **Maintainable**: Clean separation of concerns
- **Production-ready**: Includes error handling, logging, and health checks

Ready to start building?
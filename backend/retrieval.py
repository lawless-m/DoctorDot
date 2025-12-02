import duckdb
from pathlib import Path
from typing import List, Dict, Optional
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
        self.embedding_dim: Optional[int] = None

    def connect(self):
        """Establish connection to DuckDB."""
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB at {self.db_path}")

        # Install and load VSS extension
        self.conn.execute("INSTALL vss;")
        self.conn.execute("LOAD vss;")

        # Enable experimental HNSW persistence
        self.conn.execute("SET hnsw_enable_experimental_persistence = true;")

    def initialize_schema(self, embedding_dimension: int):
        """
        Create the schema for storing document chunks and embeddings.

        Args:
            embedding_dimension: Dimensionality of embedding vectors
        """
        self.embedding_dim = embedding_dimension
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS documents (
                chunk_id VARCHAR PRIMARY KEY,
                document_name VARCHAR NOT NULL,
                page_number INTEGER,
                chunk_index INTEGER NOT NULL,
                chunk_text TEXT NOT NULL,
                embedding FLOAT[{embedding_dimension}] NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

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

        # Get embedding dimension if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(query_embedding)

        # DuckDB VSS search with cosine similarity
        results = self.conn.execute(f"""
            SELECT
                chunk_id,
                document_name,
                page_number,
                chunk_index,
                chunk_text,
                array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) as similarity
            FROM documents
            WHERE array_cosine_similarity(embedding, ?::FLOAT[{self.embedding_dim}]) >= ?
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

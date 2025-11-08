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

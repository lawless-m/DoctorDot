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

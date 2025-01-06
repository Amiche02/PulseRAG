import asyncio
import logging
from ragutils.services.segment import CustomSegment
from ragutils.services.embedder import EmbeddingService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Indexer:
    """
    Automates indexing by segmenting text, embedding chunks, and organizing data for storage.
    Handles multiple documents asynchronously for high performance.
    """

    def __init__(self, segmenter: CustomSegment, embedder: EmbeddingService):
        """
        Initializes the Indexer with a segmenter and embedder.
        Args:
            segmenter (CustomSegment): An instance of the segmentation service.
            embedder (EmbeddingService): An instance of the embedding service.
        """
        self.segmenter = segmenter
        self.embedder = embedder

    async def _process_document(self, document_id: str, text: str, metadata: dict) -> dict:
        """
        Asynchronously processes a single document.
        Args:
            document_id (str): Unique identifier for the document.
            text (str): Full text of the document.
            metadata (dict): Metadata for the document (e.g., title, tags).

        Returns:
            dict: Indexed data containing chunks, embeddings, and metadata.
        """
        logger.info(f"Processing document: {document_id}")

        # Step 1: Segment the text into chunks
        chunks = self.segmenter.hybrid_segmentation(text)
        logger.info(f"Document {document_id} segmented into {len(chunks)} chunks.")

        # Step 2: Generate embeddings for each chunk asynchronously
        embeddings = await self.embedder.get_embeddings(chunks)
        logger.info(f"Generated embeddings for document {document_id}.")

        # Step 3: Organize data for storage
        indexed_data = {
            "document_id": document_id,
            "chunks": [
                {
                    "chunk_index": idx,
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": metadata,
                }
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ],
        }

        logger.info(f"Indexing complete for document: {document_id}")
        return indexed_data

    async def index_documents(self, documents: list) -> list:
        """
        Asynchronously indexes multiple documents.
        Args:
            documents (list): List of documents, where each document is a dict containing:
                - document_id (str): Unique identifier for the document.
                - text (str): Full text of the document.
                - metadata (dict): Metadata for the document.

        Returns:
            list: List of indexed data for all documents.
        """
        tasks = [
            self._process_document(doc["document_id"], doc["text"], doc["metadata"])
            for doc in documents
        ]
        return await asyncio.gather(*tasks)
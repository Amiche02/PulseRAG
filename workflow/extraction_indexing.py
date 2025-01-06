import logging
import asyncio
from typing import List, Dict

# Import services from ragutils.services
from ragutils.services import TextExtractor, CustomSegment, EmbeddingService, Indexer

logger = logging.getLogger(__name__)

class ExtractionIndexingWorkflow:
    """
    Workflow for extracting text from documents, generating embeddings, 
    and indexing each document with optional GPU support.
    """
    def __init__(
        self,
        extractor: TextExtractor = None,
        segmenter: CustomSegment = None,
        embedder: EmbeddingService = None,
    ):
        # Allow dependency injection or default to new instances
        self.extractor = extractor if extractor else TextExtractor()
        self.segmenter = segmenter if segmenter else CustomSegment()
        self.embedder = embedder if embedder else EmbeddingService()
        self.indexer = Indexer(segmenter=self.segmenter, embedder=self.embedder)

    async def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process multiple documents concurrently:
          - Extract text from each document
          - Index them (segment + embeddings)
        
        Args:
            documents (List[Dict]): A list of documents where each dict has:
                {
                  "document_id": str,
                  "file_path": str,
                  "metadata": dict
                }
        
        Returns:
            List[Dict]: A list of indexed data for all documents.
        """
        # Create a coroutine for each document
        tasks = [self._process_single_document(doc) for doc in documents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Optionally handle exceptions inside results (if you want to catch them individually)
        final_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception in process_documents: {result}")
            else:
                final_results.append(result)
        return final_results

    async def _process_single_document(self, document: Dict) -> Dict:
        """
        Extract and index a single document. 
        This helper method is used by process_documents.
        """
        doc_id = document["document_id"]
        file_path = document["file_path"]
        metadata = document.get("metadata", {})

        logger.info(f"Starting extraction and indexing for document: {doc_id}")

        # 1) Extract text from file
        extraction_result = await self.extractor.extract_text(file_path)
        logger.info(f"Extraction succeeded for document: {doc_id}")

        # 2) Index the document (segments + embeddings)
        indexed_data = await self.indexer._process_document(
            document_id=doc_id,
            text=extraction_result.text or "",
            metadata=metadata
        )
        
        logger.info(f"Document {doc_id} indexed successfully with {len(indexed_data['chunks'])} chunks.")
        return indexed_data

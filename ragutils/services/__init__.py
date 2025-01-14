from .embedder import EmbeddingService, EmbeddingModel
from .indexer import Indexer
from .segment import CustomSegment
from .text_extractor import TextExtractor, ExtractionResult

# UPDATED import to match the class rename
from .web_search import (
    WebSearchService,        # <--- Renamed from BaseWebSearchService
    DuckDuckGoSearchService,
    LangChainWebSearchService
)

__all__ = [
    "EmbeddingService",
    "EmbeddingModel",
    "Indexer",
    "CustomSegment",
    "TextExtractor",
    "ExtractionResult",
    "WebSearchService",       # <--- Make sure it's here
    "DuckDuckGoSearchService",
    "LangChainWebSearchService",
]

import os
import logging
from typing import Dict, Any, Optional
import pdfplumber
from bs4 import BeautifulSoup
import chardet
from pydantic import BaseModel
from config.config import TEXT_EXTRACTOR_CONFIG

import asyncio
from concurrent.futures import ThreadPoolExecutor

import sys
import json_log_formatter

# Configure structured logging with JSON formatter
formatter = json_log_formatter.JSONFormatter()
json_handler = logging.StreamHandler(sys.stdout)
json_handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(json_handler)
logger.setLevel(logging.INFO)

class ExtractionResult(BaseModel):
    """
    Model to represent the result of a text extraction process.

    Attributes:
        text (Optional[str]): Extracted textual content.
        tables (Optional[list]): Extracted tables from the document.
        metadata (Optional[Dict[str, Any]]): Metadata extracted from the document.
    """
    text: Optional[str] = None
    tables: Optional[list] = None
    metadata: Optional[Dict[str, Any]] = None

class TextExtractor:
    """
    Service to extract text, tables, and metadata from various file types.

    Supported file types include PDF, TXT, MD, and HTML.
    """
    def __init__(self, config=TEXT_EXTRACTOR_CONFIG):
        """
        Initializes the TextExtractor with the given configuration.

        Args:
            config (TextExtractorConfig): Configuration settings for extraction.
        """
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        logger.info("TextExtractor initialized with configuration: %s", self.config)

    async def extract_text(self, file_path: str) -> ExtractionResult:
        """
        Asynchronously extracts text from the given file based on its type.

        Args:
            file_path (str): Path to the file.

        Returns:
            ExtractionResult: Extracted text, tables, and metadata.
        """
        if not os.path.exists(file_path):
            logger.error("File not found: %s", file_path)
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._detect_file_type(file_path)
        logger.info("Detected file type: %s for file: %s", file_type, file_path)

        extraction_method_name = self.config.extraction_settings.extraction_methods.get(file_type)
        if not extraction_method_name:
            logger.error("No extraction method defined for file type: %s", file_type)
            raise ValueError(f"No extraction method defined for file type: {file_type}")

        extraction_method = getattr(self, extraction_method_name, None)
        if not extraction_method:
            logger.error("Extraction method '%s' not implemented.", extraction_method_name)
            raise NotImplementedError(f"Extraction method '{extraction_method_name}' not implemented.")

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, extraction_method, file_path
            )
            logger.info("Successfully extracted data from: %s", file_path)
            return ExtractionResult(**result)
        except Exception as e:
            logger.error("Failed to extract text from %s: %s", file_path, str(e))
            raise e

    def _detect_file_type(self, file_path: str) -> str:
        """
        Detects the file type based on its extension.

        Args:
            file_path (str): Path to the file.

        Returns:
            str: Detected file extension.
        """
        _, extension = os.path.splitext(file_path)
        return extension.lower().strip(".")

    def extract_text_from_pdf(self, file_path: str) -> dict:
        """
        Extracts text and simple tables from a PDF file.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            dict: Extracted text, tables, and metadata.
        """
        text = ""
        tables = []
        metadata = {}
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = pdf.metadata
                for page_number, page in enumerate(pdf.pages, start=1):
                    page_text = page.extract_text() or ""
                    text += page_text + "\n"
                    table = page.extract_table()
                    if table:
                        tables.append({"page": page_number, "table": table})
            return {"text": text.strip(), "tables": tables, "metadata": metadata}
        except Exception as e:
            logger.error("Error extracting PDF: %s", str(e))
            raise e

    def extract_text_from_text(self, file_path: str) -> dict:
        """
        Extracts text from a TXT or MD file.

        Args:
            file_path (str): Path to the text file.

        Returns:
            dict: Extracted text.
        """
        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                detected_encoding = chardet.detect(raw_data)["encoding"] or "utf-8"
                logger.info("Detected encoding: %s for file: %s", detected_encoding, file_path)

            # Fallback to 'utf-8' with replacement on decoding errors
            with open(file_path, "r", encoding=detected_encoding, errors="replace") as f:
                text = f.read()
            logger.info("Successfully extracted text from file: %s", file_path)
            return {"text": text.strip()}
        except Exception as e:
            logger.error("Error extracting text from %s: %s", file_path, str(e))
            raise e

    def extract_text_from_html(self, file_path: str) -> dict:
        """
        Extracts text and simple tables from an HTML file.

        Args:
            file_path (str): Path to the HTML file.

        Returns:
            dict: Extracted text, tables, and metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text(separator="\n").strip()
            tables = []
            metadata = {"title": soup.title.string if soup.title else ""}

            for table in soup.find_all("table"):
                rows = []
                for row in table.find_all("tr"):
                    cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                    rows.append(cells)
                tables.append(rows)

            return {"text": text, "tables": tables, "metadata": metadata}
        except Exception as e:
            logger.error("Error extracting HTML from %s: %s", file_path, str(e))
            raise e

from pydantic import BaseModel, Field
from typing import Optional, List

class VoiceConfig(BaseModel):
    """
    Configuration model for a voice.

    Attributes:
        name (str): Name of the voice.
        language (str): Language code supported by the voice.
        model_path (str): Path or identifier for the TTS model.
        speaker_id (Optional[str]): Identifier for multi-speaker models.
    """
    name: str
    language: str
    model_path: str
    speaker_id: Optional[str] = None

class EmbeddingModelConfig(BaseModel):
    """
    Configuration model for an embedding model.

    Attributes:
        name (str): Name of the embedding model.
        language (Optional[List[str]]): Languages supported by the model.
        model_path (str): Path or identifier for the SentenceTransformer model.
        description (Optional[str]): Description of the model.
    """
    name: str
    language: Optional[List[str]] = None
    model_path: str
    description: Optional[str] = None

class FileTypeConfig(BaseModel):
    """
    Configuration model for supported file types and their extraction methods.

    Attributes:
        supported_extensions (List[str]): List of supported file extensions.
        extraction_methods (dict): Mapping from file extensions to extraction methods.
    """
    supported_extensions: List[str]
    extraction_methods: dict  

class TextExtractorConfig(BaseModel):
    """
    Configuration model for the TextExtractor service.

    Attributes:
        extraction_settings (FileTypeConfig): Settings related to file extraction.
        temp_upload_dir (str): Directory to store temporary file uploads.
    """
    extraction_settings: FileTypeConfig
    temp_upload_dir: str = Field("./temp_uploads", description="Directory to store temporary file uploads.")

# Define the configuration for text extraction
TEXT_EXTRACTOR_CONFIG = TextExtractorConfig(
    extraction_settings=FileTypeConfig(
        supported_extensions=["pdf", "txt", "md", "html"],
        extraction_methods={
            "pdf": "extract_text_from_pdf",
            "txt": "extract_text_from_text",
            "md": "extract_text_from_text",
            "html": "extract_text_from_html"
        }
    )
)

# Define available voices for TTS
AVAILABLE_VOICES = [
    {
        "name": "en-US-Standard",
        "language": "en",
        "model_path": "tts_models/en/ljspeech/tacotron2-DDC",
    },
    {
        "name": "fr-FR-Standard",
        "language": "fr",
        "model_path": "tts_models/fr/mai/tacotron2-DDC",
    },
]

# Define available embedding models
AVAILABLE_EMBEDDING_MODELS = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "language": ["en", "fr", "it", "es", "de", "zh", "ja", "ru", "ar"],
        "model_path": "paraphrase-multilingual-MiniLM-L12-v2",
        "description": "A multilingual model supporting multiple languages."
    },
    {
        "name": "all-MiniLM-L12-v2",
        "language": ["en"],
        "model_path": "all-MiniLM-L12-v2",
        "description": "An English-only model optimized for speed and accuracy."
    },
]

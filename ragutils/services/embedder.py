import asyncio
import logging
from typing import List, Optional, Dict
from langdetect import detect, DetectorFactory, LangDetectException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor

from config.config import AVAILABLE_EMBEDDING_MODELS, EmbeddingModelConfig

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingModel(BaseModel):
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

class EmbeddingService:
    """
    Service to manage and generate embeddings using multiple SentenceTransformer models.

    Handles model selection based on input language and manages model loading and addition.
    """
    def __init__(self):
        """
        Initializes the EmbeddingService by loading available models and preloading them.
        """
        self.models: Dict[str, SentenceTransformer] = {}
        self.embedding_models: List[EmbeddingModel] = self._load_available_models()
        self._initialize_models()

        # Initialize thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=torch.cuda.device_count() or 4)

    def _load_available_models(self) -> List[EmbeddingModel]:
        """
        Loads available embedding models from the configuration.

        Returns:
            List[EmbeddingModel]: A list of configured embedding models.
        """
        embedding_models = []
        for model_dict in AVAILABLE_EMBEDDING_MODELS:
            try:
                model = EmbeddingModel(**model_dict)
                embedding_models.append(model)
                logger.info(f"Loaded embedding model configuration: {model.name}")
            except Exception as e:
                logger.error(f"Error loading embedding model configuration {model_dict}: {str(e)}")
        return embedding_models

    def _initialize_models(self):
        """
        Preloads all embedding models to optimize performance during runtime.
        """
        for model in self.embedding_models:
            try:
                logger.info(f"Loading SentenceTransformer model: {model.name}")
                st_model = SentenceTransformer(model.model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
                self.models[model.name] = st_model
                logger.info(f"Loaded model: {model.name}")
            except Exception as e:
                logger.error(f"Failed to load SentenceTransformer model {model.name}: {str(e)}")

    def list_models(self) -> List[EmbeddingModel]:
        """
        Retrieves the list of available embedding models.

        Returns:
            List[EmbeddingModel]: Available embedding models.
        """
        return self.embedding_models

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the provided text.

        Args:
            text (str): The input text.

        Returns:
            str: Detected language code.

        Raises:
            ValueError: If language detection fails.
        """
        try:
            language = detect(text)
            logger.info(f"Detected language: {language}")
            return language
        except LangDetectException as e:
            logger.error(f"Language detection failed: {str(e)}")
            raise ValueError("Could not detect language of the input text.")

    def select_best_model(self, language: Optional[str] = None) -> SentenceTransformer:
        """
        Selects the most appropriate embedding model based on the detected language.

        Args:
            language (Optional[str]): Detected language code. If None, uses the default model.

        Returns:
            SentenceTransformer: The selected embedding model.

        Raises:
            ValueError: If no suitable model is found.
        """
        if language:
            suitable_models = [
                model for model in self.embedding_models
                if model.language and language in model.language
            ]
            if suitable_models:
                # Prioritize models supporting more languages
                suitable_models.sort(key=lambda x: len(x.language), reverse=True)
                selected_model = suitable_models[0]
                logger.info(f"Selected model '{selected_model.name}' for language '{language}'")
                return self.models[selected_model.name]
            else:
                logger.warning(f"No specific model found for language '{language}'. Using default model.")

        if self.embedding_models:
            default_model = self.embedding_models[0]
            logger.info(f"Using default model '{default_model.name}'")
            return self.models[default_model.name]
        else:
            logger.error("No embedding models are available.")
            raise ValueError("No embedding models are available.")

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts using the appropriate models with GPU optimization and batching.

        Args:
            texts (List[str]): List of texts to generate embeddings for.

        Returns:
            List[List[float]]: List of embeddings corresponding to each input text.

        Raises:
            ValueError: If input validation fails.
        """
        if not texts or not all(isinstance(text, str) and text.strip() for text in texts):
            return []

        # Group texts by detected language
        language_groups: Dict[str, List[str]] = {}
        for text in texts:
            try:
                language = self.detect_language(text)
                language_groups.setdefault(language, []).append(text)
            except ValueError as ve:
                logger.warning(f"Skipping text due to language detection error: {ve}")

        embeddings = []

        async def process_batch(batch_texts: List[str], model: SentenceTransformer):
            """Processes a batch of texts asynchronously."""
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.executor, lambda: model.encode(batch_texts, show_progress_bar=False, batch_size=32))

        for language, grouped_texts in language_groups.items():
            model = self.select_best_model(language)
            logger.info(f"Processing {len(grouped_texts)} texts with model '{model}.")
            batch_size = 32
            for i in range(0, len(grouped_texts), batch_size):
                batch = grouped_texts[i:i + batch_size]
                try:
                    batch_embeddings = await process_batch(batch, model)
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")

        return embeddings

    def add_embedding_model(self, model_config: EmbeddingModelConfig):
        """
        Adds a new embedding model to the service dynamically.

        Args:
            model_config (EmbeddingModelConfig): Configuration for the new embedding model.

        Raises:
            ValueError: If the model cannot be loaded.
        """
        try:
            logger.info(f"Adding new embedding model: {model_config.name}")
            st_model = SentenceTransformer(model_config.model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
            self.models[model_config.name] = st_model
            self.embedding_models.append(model_config)
            logger.info(f"Successfully added embedding model: {model_config.name}")
        except Exception as e:
            logger.error(f"Failed to add embedding model {model_config.name}: {str(e)}")
            raise ValueError(f"Failed to add embedding model {model_config.name}: {str(e)}")

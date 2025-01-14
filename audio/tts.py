import asyncio
import logging
from typing import List, Optional, Dict, AsyncGenerator
from langdetect import detect, DetectorFactory, LangDetectException
from pydantic import BaseModel
from TTS.api import TTS
import torch
import io
import soundfile as sf  # <-- Added for in-memory WAV conversion

from config.config import AVAILABLE_VOICES, VoiceConfig

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Voice(BaseModel):
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

class TTSService:
    """
    Service to manage and perform text-to-speech (TTS) operations.

    Handles language detection, voice selection, model loading, and speech synthesis.
    """
    def __init__(self):
        """
        Initializes the TTSService by loading available voices and preloading their models.
        """
        self.voices: List[Voice] = self._load_available_voices()
        self.tts_models: Dict[str, TTS] = {}
        self._initialize_models()

    def _load_available_voices(self) -> List[Voice]:
        """
        Loads available voices from the configuration.

        Returns:
            List[Voice]: A list of configured voices.
        """
        voices = []
        for voice_dict in AVAILABLE_VOICES:
            try:
                voice = Voice(**voice_dict)
                voices.append(voice)
                logger.info(f"Loaded voice configuration: {voice.name}")
            except Exception as e:
                logger.error(f"Error loading voice configuration {voice_dict}: {str(e)}")
        return voices

    def _initialize_models(self):
        """
        Preloads TTS models for all available voices to optimize performance.
        """
        for voice in self.voices:
            try:
                logger.info(f"Loading TTS model for voice: {voice.name}")
                gpu_available = torch.cuda.is_available()
                tts = TTS(model_name=voice.model_path, progress_bar=False, gpu=gpu_available)
                self.tts_models[voice.name] = tts
                logger.info(f"Loaded model for voice: {voice.name}")
            except Exception as e:
                logger.error(f"Failed to load model for voice {voice.name}: {str(e)}")

    def list_voices(self) -> List[Voice]:
        """
        Retrieves the list of available voices.

        Returns:
            List[Voice]: Available voices.
        """
        return self.voices

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

    def select_voice(self, language: str) -> Voice:
        """
        Selects an appropriate voice based on the detected language.

        Args:
            language (str): Detected language code.

        Returns:
            Voice: The selected voice configuration.

        Raises:
            ValueError: If no suitable voice is found for the language.
        """
        for voice in self.voices:
            if voice.language == language:
                logger.info(f"Selected voice: {voice.name} for language: {language}")
                return voice
        logger.error(f"No available voice for language: {language}")
        raise ValueError(f"No available voice for language: {language}")
    
    async def synthesize_speech_stream(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Converts text to speech and yields audio chunks as they are generated.

        Args:
            text (str): The input text to synthesize.
            voice_name (Optional[str]): Specific voice to use. If None, selects based on language.

        Yields:
            bytes: Audio data chunks in WAV format.

        Raises:
            ValueError: If the specified voice is not found or model is not loaded.
            RuntimeError: If speech synthesis fails.
        """
        if voice_name:
            voice = next((v for v in self.voices if v.name == voice_name), None)
            if not voice:
                logger.error(f"Voice '{voice_name}' not found.")
                raise ValueError(f"Voice '{voice_name}' not found.")
        else:
            language = self.detect_language(text)
            voice = self.select_voice(language)

        tts = self.tts_models.get(voice.name)
        if not tts:
            logger.error(f"TTS model for voice '{voice.name}' is not loaded.")
            raise ValueError(f"TTS model for voice '{voice.name}' is not loaded.")

        logger.info(f"Synthesizing speech for voice: {voice.name}")
        try:
            # Coqui TTS automatically splits text into sentences internally for you,
            # or you can do it manually. We'll keep it simple here:
            # Get raw audio samples for the entire text at once
            loop = asyncio.get_event_loop()
            audio_samples = await loop.run_in_executor(None, lambda: tts.tts(text))

            # Convert to WAV in memory
            sample_rate = tts.synthesizer.output_sample_rate
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio_samples, sample_rate, format="WAV")
                wav_buffer.seek(0)
                yield wav_buffer.read()

        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            raise RuntimeError("Speech synthesis failed.")

    async def synthesize_sentence(self, sentence: str, tts_model: TTS) -> bytes:
        """
        Synthesizes speech for a single sentence (helper method for a chunked approach).
        """
        loop = asyncio.get_event_loop()
        audio_samples = await loop.run_in_executor(None, lambda: tts_model.tts(sentence))

        # Convert the raw audio samples to WAV bytes
        sample_rate = tts_model.synthesizer.output_sample_rate
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, audio_samples, sample_rate, format="WAV")
            wav_buffer.seek(0)
            audio_bytes = wav_buffer.read()
        return audio_bytes

    async def synthesize_speech(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> bytes:
        """
        Converts text to speech using the specified or default voice, returning WAV bytes.

        Args:
            text (str): The input text to synthesize.
            voice_name (Optional[str]): Specific voice to use. If None, detects language and picks best voice.

        Returns:
            bytes: Audio data in WAV format.

        Raises:
            ValueError: If the specified voice is not found or model is not loaded.
            RuntimeError: If speech synthesis fails.
        """
        # Determine appropriate voice
        if voice_name:
            voice = next((v for v in self.voices if v.name == voice_name), None)
            if not voice:
                logger.error(f"Voice '{voice_name}' not found.")
                raise ValueError(f"Voice '{voice_name}' not found.")
        else:
            language = self.detect_language(text)
            voice = self.select_voice(language)

        # Check if model is loaded
        tts = self.tts_models.get(voice.name)
        if not tts:
            logger.error(f"TTS model for voice '{voice.name}' is not loaded.")
            raise ValueError(f"TTS model for voice '{voice.name}' is not loaded.")

        logger.info(f"Synthesizing speech for voice: {voice.name}")
        try:
            loop = asyncio.get_event_loop()
            audio_samples = await loop.run_in_executor(None, lambda: tts.tts(text))

            # Convert samples to WAV in memory
            sample_rate = tts.synthesizer.output_sample_rate
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, audio_samples, sample_rate, format="WAV")
                wav_buffer.seek(0)
                audio_bytes = wav_buffer.read()

            logger.info("Speech synthesis successful.")
            return audio_bytes

        except Exception as e:
            logger.error(f"Speech synthesis failed: {str(e)}")
            raise RuntimeError("Speech synthesis failed.")

# Provide a single service instance if you want a shared instance
tts_service = TTSService()

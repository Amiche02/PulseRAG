import logging
import asyncio
from typing import Optional, AsyncGenerator

from audio import TTSService
from workflow.tools.audio_player import play_wav_bytes

logger = logging.getLogger(__name__)

class TTSGenerationWorkflow:
    """
    Demonstrates an asynchronous workflow for text-to-speech generation.
    Leverages TTSService for voice selection and GPU usage (if available).
    """

    def __init__(self, tts_service: TTSService = None):
        self.tts_service = tts_service if tts_service else TTSService()

    async def synthesize_speech(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> bytes:
        """
        Synthesize the entire text into a single audio output (WAV).

        Args:
            text (str): The text to be synthesized.
            voice_name (Optional[str]): The name of a specific voice to use.

        Returns:
            bytes: The audio data in WAV format.
        """
        logger.info(f"Synthesizing TTS for text of length {len(text)} with voice={voice_name}")
        audio_bytes = await self.tts_service.synthesize_speech(text, voice_name=voice_name)
        return audio_bytes

    async def synthesize_and_play(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> None:
        """
        Synthesize the text and immediately play it using an audio player (e.g., playsound).

        Args:
            text (str): The text to be synthesized.
            voice_name (Optional[str]): The name of a specific voice to use.

        Note:
            This does not save the audio to disk (other than a temporary file used for playback).
        """
        logger.info(f"Synthesizing TTS (immediate playback) for text of length {len(text)}")
        audio_bytes = await self.synthesize_speech(text, voice_name=voice_name)
        logger.info("Playing audio now...")
        play_wav_bytes(audio_bytes)

    async def synthesize_speech_stream(
        self,
        text: str,
        voice_name: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Streams audio chunks in WAV format as they're generated.

        Args:
            text (str): The text to be synthesized.
            voice_name (Optional[str]): Name of the specific voice to use.

        Yields:
            bytes: Audio data chunk in WAV format.
        """
        logger.info(f"Streaming TTS for text of length {len(text)} with voice={voice_name}")
        async for audio_chunk in self.tts_service.synthesize_speech_stream(text, voice_name=voice_name):
            yield audio_chunk

import asyncio
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from workflow import (
    ExtractionIndexingWorkflow,
    TTSGenerationWorkflow
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    # 1) Extraction and Indexing
    extraction_workflow = ExtractionIndexingWorkflow()

    documents = [
        {
            "document_id": "doc1",
            "file_path": "/home/amiche/Downloads/french-time-merchant.pdf",
            "metadata": {"title": "Document 1"}
        },
        {
            "document_id": "doc2",
            "file_path": "/home/amiche/Downloads/MyWife.md",
            "metadata": {"title": "Document 2"}
        }
    ]
    indexed_docs = await extraction_workflow.process_documents(documents)
    logger.info(f"Indexed documents: {indexed_docs}")

    # 2) Text-to-Speech
    tts_workflow = TTSGenerationWorkflow()

    # 1) English TTS example: Synthesize and save
    text_to_speak_en = "Hello from our TTS service. This file will be saved locally."
    audio_data_en = await tts_workflow.synthesize_speech(text_to_speak_en, voice_name="en-US-Standard")
    with open("hello_en.wav", "wb") as f:
        f.write(audio_data_en)
    logger.info("English audio saved to hello_en.wav")

    # 2) English TTS example: Synthesize and play (no local file saved)
    text_to_play_en = "This is the English TTS playback, not saving to disk."
    await tts_workflow.synthesize_and_play(text_to_play_en, voice_name="en-US-Standard")

    # 3) French TTS example: Synthesize and save
    text_to_speak_fr = "Bonjour de notre service TTS. Ce fichier sera enregistré localement."
    audio_data_fr = await tts_workflow.synthesize_speech(text_to_speak_fr, voice_name="fr-FR-Standard")
    with open("hello_fr.wav", "wb") as f:
        f.write(audio_data_fr)
    logger.info("French audio saved to hello_fr.wav")

    # 4) French TTS example: Synthesize and play (no local file saved)
    text_to_play_fr = "Ceci est la version française, et nous ne sauvegardons pas ce fichier."
    await tts_workflow.synthesize_and_play(text_to_play_fr, voice_name="fr-FR-Standard")


if __name__ == "__main__":
    asyncio.run(main())

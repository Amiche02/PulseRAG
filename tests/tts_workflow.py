import argparse
import asyncio
import logging
import os
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from workflow import TTSGenerationWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Run TTS workflow for given text.")
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to synthesize speech from."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="fr-FR-Standard",
        help="Optional: Specify voice name (e.g., 'en-US-Standard' or 'fr-FR-Standard')."
    )
    parser.add_argument(
        "--mode",
        choices=["save", "play"],
        default="play",
        help="Mode of operation: 'save' to save audio to a file, 'play' to stream playback."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="output.wav",
        help="Output file path when saving audio."
    )
    return parser.parse_args()

async def main():
    args = parse_args()

    tts_workflow = TTSGenerationWorkflow()
    voice_name = args.voice
    text = args.text

    if args.mode == "save":
        audio_data = await tts_workflow.synthesize_speech(text, voice_name=voice_name)
        with open(args.output_file, "wb") as f:
            f.write(audio_data)
        logger.info(f"Audio saved to {args.output_file}")
    else:  # play mode
        await tts_workflow.synthesize_and_play(text, voice_name=voice_name)
        logger.info("Playback completed.")

if __name__ == "__main__":
    asyncio.run(main())

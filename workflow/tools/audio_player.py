# workflow/tools/audio_player.py

import os
import tempfile
import logging

import simpleaudio  # Instead of playsound

logger = logging.getLogger(__name__)

def play_wav_bytes(audio_data: bytes):
    """
    Plays WAV audio data in-memory. Because many libraries expect a file path,
    we temporarily write to a .wav file, then use simpleaudio to play it.

    This avoids needing PyGObject / GStreamer libraries on Linux.
    """
    if not audio_data:
        logger.error("No audio data to play.")
        return

    # Create a temporary WAV file to feed to simpleaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_data)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # Load and play the WAV file
        wave_obj = simpleaudio.WaveObject.from_wave_file(tmp_path)
        play_obj = wave_obj.play()

        # Wait until playback is finished
        play_obj.wait_done()

    except Exception as e:
        logger.error(f"Error playing audio: {e}")

    finally:
        # Remove the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

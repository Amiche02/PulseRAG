import sys
import io
import queue
import logging
import requests
import wave
import struct

import sounddevice as sd
import webrtcvad

from config.config import STTConfig

logger = logging.getLogger(__name__)

class STTService:
    """
    A speech-to-text service that captures microphone audio in real-time,
    uses WebRTC VAD to detect speech segments, and sends them to whisper.cpp.
    """

    def __init__(self, stt_config: STTConfig = STTConfig()):
        """
        Args:
            stt_config (STTConfig): Configuration object from config.config.
        """
        self.config = stt_config

        self.server_url = self.config.server_url
        self.language = self.config.language
        self.sample_rate = self.config.sample_rate

        # WebRTC VAD Setup
        self.vad = webrtcvad.Vad()
        self.vad.set_mode(self.config.vad_aggressiveness)  # 0-3

        # Frame sizes
        self.chunk_ms = self.config.chunk_ms
        # must be 10, 20, or 30 ms
        assert self.chunk_ms in (10, 20, 30), "chunk_ms must be 10, 20, or 30"
        self.frame_bytes = int(self.sample_rate * 2 * self.chunk_ms / 1000)
        # Explanation:
        #   sample_rate * 2 bytes/sample * chunk_ms/1000 => # of bytes per chunk

        # Silence
        self.silence_ms = self.config.silence_ms
        self.silence_frames = int(self.silence_ms / self.chunk_ms)

        # We'll store frames in a queue
        self.audio_queue = queue.Queue()
        self.is_listening = False

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback. We receive raw bytes from the mic."""
        if status:
            logger.warning(f"STTService callback status: {status}")
        # Put the raw PCM data into the queue
        self.audio_queue.put(bytes(indata))

    def listen_infinite(self):
        """
        Infinite loop capturing from mic, chunking audio into frames of 30ms,
        feeding them into webrtcvad, and finalizing segments after enough silence.
        """
        self.is_listening = True

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16',
            callback=self._audio_callback,
            blocksize=1024
        ):
            logger.info("STTService: Microphone stream opened. Listening with WebRTC VAD...")

            # Buffer for partial frames that we haven't yet formed into a full chunk
            leftover_bytes = b""

            # The chunk that represents the entire "voiced segment"
            speech_buffer = []

            # Counters for silence detection
            num_silent_frames = 0
            in_speech = False

            try:
                while self.is_listening:
                    try:
                        data = self.audio_queue.get(timeout=1.0)
                    except queue.Empty:
                        continue

                    # Append to leftover
                    leftover_bytes += data

                    # While leftover has enough bytes for one 30ms frame:
                    while len(leftover_bytes) >= self.frame_bytes:
                        frame = leftover_bytes[:self.frame_bytes]
                        leftover_bytes = leftover_bytes[self.frame_bytes:]

                        # Check if frame is speech:
                        is_speech = self.vad.is_speech(frame, self.sample_rate)

                        if is_speech:
                            speech_buffer.append(frame)
                            if not in_speech:
                                in_speech = True
                            num_silent_frames = 0
                        else:
                            if in_speech:
                                # We were in speech, now we got silence
                                num_silent_frames += 1
                                speech_buffer.append(frame)

                                # If we've gone enough silent frames => finalize
                                if num_silent_frames >= self.silence_frames:
                                    # Finalize the entire speech_buffer
                                    audio_data = b"".join(speech_buffer)
                                    transcription = self.transcribe_chunk(audio_data)
                                    logger.info(f"[TRANSCRIBED] {transcription.strip()}")
                                    print(f"\n** You said: {transcription.strip()}\n")

                                    # reset
                                    speech_buffer.clear()
                                    in_speech = False
                                    num_silent_frames = 0
                            else:
                                # We are not in_speech, do nothing
                                pass

            except KeyboardInterrupt:
                logger.info("STTService: KeyboardInterrupt -> stopping.")
            except Exception as e:
                logger.error(f"STTService error: {e}", exc_info=True)
            finally:
                self.is_listening = False

    def transcribe_chunk(self, audio_data: bytes) -> str:
        """
        Convert the entire chunk (which is a series of 30ms frames) to a WAV,
        then send to whisper.cpp server for transcription.
        """
        wav_bytes = self._convert_pcm_to_wav(audio_data)
        files = {
            'file': ('chunk.wav', wav_bytes, 'audio/wav'),
        }
        data = {
            "language": self.language,
            "response_format": "json",
            # e.g. "translate": "false",
            # "no_timestamps": "true",
        }
        try:
            resp = requests.post(self.server_url, files=files, data=data, timeout=300)
            resp.raise_for_status()
            result_json = resp.json()
            return result_json.get("text", "")
        except Exception as ex:
            logger.error(f"Error posting to whisper.cpp server: {ex}")
            return "[ERROR: no transcription]"

    def _convert_pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """
        Convert raw 16-bit PCM data to WAV. sample_rate is self.sample_rate, 1 channel.
        """
        import io
        with io.BytesIO() as wav_buffer:
            with wave.Wave_write(wav_buffer) as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(self.sample_rate)
                wf.writeframes(pcm_data)
            return wav_buffer.getvalue()

    def stop_listening(self):
        """Stop the infinite loop gracefully."""
        self.is_listening = False

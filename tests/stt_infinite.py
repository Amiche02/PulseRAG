import logging
import sys
import os

sys.path.append(os.path.abspath(os.path.join(__file__, "../../")))

from audio import STTService
from config.config import STTConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    stt_config = STTConfig(
        server_url="http://127.0.0.1:8080/inference",
        language="fr",
        vad_aggressiveness=1,  # 0..3
        chunk_ms=30,           # choose 10, 20, or 30
        silence_ms=1200         # 0.8 sec of silence to finalize
    )
    stt_service = STTService(stt_config)
    logger.info("Starting infinite STT with WebRTC VAD. Ctrl+C to exit.")
    stt_service.listen_infinite()

if __name__ == "__main__":
    main()

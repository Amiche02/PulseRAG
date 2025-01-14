# PulseRAG: Retrieval-Augmented Generation for Hello Pulse

**PulseRAG** is a core component of **Hello Pulse**, designed to handle document processing, web search integration, text embeddings, and text-to-speech (TTS) functionalities to enrich brainstorming and innovation sessions. It also supports speech-to-text (STT) for real-time transcription. This module seamlessly transforms external knowledge into actionable insights via Retrieval-Augmented Generation (RAG).

---

## ✅ Features

- **Document Extraction & Indexing:**
  - Extract text, tables, and metadata from PDFs, DOCX, TXT, HTML, etc.
  - Hybrid text segmentation with token and sentence-based chunking.
  - Supports embeddings with `sentence-transformers`.

- **Web Search Integration:**
  - Perform searches using DuckDuckGo or LangChain’s API.
  - Scrape relevant pages and embed indexed data for context-aware responses.

- **Text-to-Speech (TTS) and Speech-to-Text (STT):**
  - TTS: Generate multilingual speech with models like `Coqui TTS`.
  - STT: Real-time speech-to-text transcription using `Whisper`  ---> setup in docker/whisper.

- **Embeddings Service:**
  - Supports multiple embedding models.
  - Dynamic model loading for optimized performance.
  - Language detection for automatic model selection.

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+ (tested up to 3.11)
- CUDA-compatible GPU (optional for GPU acceleration)
- **Linux-based systems:** Install additional packages for audio and Python development.

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Amiche02/PulseRAG.git 
   cd PulseRAG
   ```

2. **Install Python and dependencies:**
   ```bash
   sudo apt update
   sudo apt install libasound2-dev
   sudo apt install ffmpeg
   sudo apt-get install portaudio19-dev
   
   sudo add-apt-repository ppa:deadsnakes/ppa       # Optional for Python 3.11
   sudo apt install python3.11-dev
   ```

3. **Install `PyTorch` (with GPU support):**
   ```bash
   pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
   ```

4. **Install project dependencies:**
   ```bash
   cd env
   pip install -r requirements.gpu.txt  # For GPU support
   # Or for CPU-only:
   pip install -r requirements.cpu.txt
   ```

5. **Install `simpleaudio` for audio playback:**
   ```bash
   pip install simpleaudio
   ```

6. **Download language models for `spaCy` (NLP support):**
   ```bash
   bash env/spacy_models.sh
   ```

---

## 🚀 Usage

### 1. Run the RAG Server
To launch the TTS, STT, and web search workflows:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. TTS Example (Text-to-Speech)
To synthesize speech:
```python
from ragutils.services.tts import TTSService

tts_service = TTSService()
audio_bytes = await tts_service.synthesize_speech("Hello from PulseRAG", voice_name="en-US-Standard")
```

### 3. STT Example (Speech-to-Text)
To transcribe audio in French:
```bash
./build/bin/whisper-command -m ./models/ggml-medium.bin -t 8 -l fr
```
Or run the Whisper server:
```bash
./build/bin/whisper-server --model ./models/ggml-large-v3-turbo.bin --threads 8 --port 8080 --host 0.0.0.0 --language fr --print-realtime --print-progress
```

### 4. Document Indexing Example
```python
from workflow.extraction_indexing import ExtractionIndexingWorkflow

documents = [
    {"document_id": "doc1", "file_path": "/path/to/doc.pdf", "metadata": {"title": "Sample Doc"}}
]
workflow = ExtractionIndexingWorkflow()
indexed_docs = await workflow.process_documents(documents)
```

---

## 📄 Project Structure

```
├── main.py           # Main entry point
├── config/           # Configuration files
├── env/              # Dependencies (CPU/GPU)
├── ragutils/         # Core services (embeddings, TTS, STT, web search, etc.)
├── workflow/         # Document extraction and indexing workflows
├── tests/            # Test scripts for web search, TTS, and STT
└── docs/             # Documentation
```

---

## 🛡️ Testing

Run the tests:
```bash
python -m unittest discover tests
```

---

## 🛣️ Notes

- **TTS and STT Models:** This project uses `Coqui TTS` and `OpenAI Whisper` for text-to-speech and speech-to-text, respectively.
- GPU is recommended for faster processing, especially when handling large texts or real-time speech transcription.

---

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 📢 Additional Information

- **Author:** O.A. Stéphane KPOVIESSI
- **Email:** oastephaneamiche@gmail.com
- **GitHub:** [Amiche02](https://github.com/Amiche02)


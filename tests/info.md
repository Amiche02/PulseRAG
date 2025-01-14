

**Usage Examples:**

- **Treat documents:**

```bash
python3.11 tests/docs_workflow.py --docs /path/to/doc1.pdf /path/to/doc2.md --save-format md --output-dir outputs/my_docs
```


- **Save Audio to a File:**

  ```bash
  python3.11 tests/tts_workflow.py --text "Hello, world!" --voice en-US-Standard --mode save --output-file hello.wav
  ```

- **Stream Playback of Audio:**

  ```bash
  python3.11 tests/tts_workflow.py --text "Bonjour!" --voice fr-FR-Standard --mode play
  ```
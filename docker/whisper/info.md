```sh
make -j medium
make -j large-v3-turbo

./build/bin/whisper-command -m ./models/ggml-medium.bin -t 8 -l fr

./build/bin/whisper-command -m ./models/ggml-large-v3-turbo.bin -t 8 -l fr

./build/bin/whisper-server --model ./models/ggml-large-v3-turbo.bin --threads 8 --port 8080 --host 0.0.0.0 --language fr --print-realtime --print-progress

```

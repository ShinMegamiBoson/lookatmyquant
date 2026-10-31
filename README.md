# LookAtMyQuant

A quantization compiler that delivers 12× lower KL than Q4_K_M at the same file size.

This repository holds the paper website. The page loads `data.json`, so open it through a local web server:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000.

Measured on Gemma 4 E2B. The compiled GGUF uses only llama.cpp's own quantization types, so it runs in llama.cpp as it is.

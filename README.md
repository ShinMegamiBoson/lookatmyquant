# LookAtMyQuant

A quantization compiler that picks a code for every piece of a model.

This repository holds the paper website. The page loads `data.json`, so open it through a local web server:

```sh
python3 -m http.server 8000
```

Then visit http://localhost:8000.

On Gemma 4 E2B, a GGUF file made by the compiler from llama.cpp's own quantization types has 12 times lower KL divergence from the original model than llama.cpp's Q4_K_M at the same size.

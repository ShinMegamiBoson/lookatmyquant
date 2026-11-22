"""Apple Silicon MLX runtime for packed Inkling-2-Mini models and 4-bit DSpark speculative decoding."""

from .drafter import DSparkDrafterMLX, quantize_mtp_to_4bit, resolve_drafter_files
from .generate import PromptCache, generate_dspark, generate_serial, stream_dspark, stream_serial
from .model import Inkling2Cache, Inkling2Config, Inkling2LoopPackedMLX, Inkling2MLXModel, load_inkling

__all__ = [
    "DSparkDrafterMLX",
    "Inkling2Cache",
    "Inkling2Config",
    "Inkling2LoopPackedMLX",
    "Inkling2MLXModel",
    "PromptCache",
    "generate_dspark",
    "generate_serial",
    "load_inkling",
    "quantize_mtp_to_4bit",
    "resolve_drafter_files",
    "stream_dspark",
    "stream_serial",
]

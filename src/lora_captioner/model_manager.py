"""
Model download and management for LoRA Captioner.

Handles downloading, caching, and loading the Florence-2-PromptGen model.
"""

import os
from pathlib import Path
from typing import Literal

import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# Default model for captioning
# Using BLIP for best compatibility and stability
DEFAULT_MODEL_ID = "Salesforce/blip-image-captioning-large"

# Cache directory for models
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "lora-captioner" / "models"


DeviceType = Literal["auto", "cuda", "cpu"]


def detect_device(requested: DeviceType = "auto") -> tuple[str, torch.dtype]:
    """
    Detect the best available device for inference.
    
    Args:
        requested: User-requested device ("auto", "cuda", or "cpu")
        
    Returns:
        Tuple of (device string, dtype)
    """
    if requested == "cpu":
        return "cpu", torch.float32
    
    if requested == "cuda" or (requested == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available():
            # Check VRAM availability
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram_gb >= 2.0:  # Florence-2-PromptGen needs ~1-2GB
                return "cuda:0", torch.float16
            else:
                print(f"Warning: Only {vram_gb:.1f}GB VRAM available. Using CPU instead.")
                return "cpu", torch.float32
        elif requested == "cuda":
            raise RuntimeError("CUDA requested but not available")
    
    return "cpu", torch.float32


def get_model_path(model_id: str = DEFAULT_MODEL_ID, cache_dir: Path | None = None) -> Path:
    """
    Get the local cache path for a model.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Custom cache directory (optional)
        
    Returns:
        Path to the model cache directory
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    # Create cache directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    return cache_dir / model_id.replace("/", "--")


def is_model_cached(model_id: str = DEFAULT_MODEL_ID, cache_dir: Path | None = None) -> bool:
    """
    Check if a model is already downloaded.
    
    Args:
        model_id: HuggingFace model ID
        cache_dir: Custom cache directory (optional)
        
    Returns:
        True if model is cached, False otherwise
    """
    model_path = get_model_path(model_id, cache_dir)
    return model_path.exists() and any(model_path.iterdir())


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: DeviceType = "auto",
    cache_dir: Path | None = None,
):
    """
    Load the captioning model and processor.
    
    Downloads the model if not already cached.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on ("auto", "cuda", or "cpu")
        cache_dir: Custom cache directory (optional)
        
    Returns:
        Tuple of (model, processor, device_string)
    """
    device_str, dtype = detect_device(device)
    
    print(f"Loading model: {model_id}")
    print(f"Device: {device_str}, dtype: {dtype}")
    
    # Load BLIP processor and model
    processor = BlipProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )
    
    model = BlipForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        cache_dir=cache_dir,
    ).to(device_str)
    
    model.eval()
    
    return model, processor, device_str

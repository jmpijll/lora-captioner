"""
Image captioning logic for LoRA Captioner.

Handles generating captions using BLIP with
different strategies for Character, Style, and Concept LoRAs.
"""

from enum import Enum
from pathlib import Path

import torch
from PIL import Image


class LoRAType(str, Enum):
    """Types of LoRA training."""
    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"


# Text prompts for conditional captioning by LoRA type
# BLIP supports conditional captioning with text prompts
LORA_TYPE_PROMPTS = {
    LoRAType.CHARACTER: "a photo of",  # Generic prompt, BLIP will describe what it sees
    LoRAType.STYLE: "an image of",
    LoRAType.CONCEPT: "a picture showing",
}


def get_prompt(lora_type: LoRAType) -> str:
    """
    Get the appropriate prompt for a LoRA type.
    
    Args:
        lora_type: Type of LoRA being trained
        
    Returns:
        Prompt string for the model
    """
    return LORA_TYPE_PROMPTS.get(lora_type, LORA_TYPE_PROMPTS[LoRAType.STYLE])


def caption_image(
    image_path: Path,
    model,
    processor,
    device: str,
    lora_type: LoRAType,
    trigger_word: str | None = None,
) -> str:
    """
    Generate a caption for a single image using BLIP.
    
    Args:
        image_path: Path to the image file
        model: Loaded BLIP model
        processor: BLIP processor
        device: Device string (e.g., "cuda:0" or "cpu")
        lora_type: Type of LoRA being trained
        trigger_word: Optional trigger word to prepend
        
    Returns:
        Generated caption string
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get conditional prompt for this LoRA type
    text_prompt = get_prompt(lora_type)
    
    # Process inputs with conditional text
    inputs = processor(image, text_prompt, return_tensors="pt").to(device)
    
    # Get model dtype and convert pixel values
    model_dtype = next(model.parameters()).dtype
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
    
    # Generate caption
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=3,
            early_stopping=True,
        )
    
    # Decode the output
    caption = processor.decode(output[0], skip_special_tokens=True)
    caption = caption.strip()
    
    # Prepend trigger word if specified
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    
    return caption


def caption_batch(
    image_paths: list[Path],
    model,
    processor,
    device: str,
    lora_type: LoRAType,
    trigger_word: str | None = None,
    progress_callback=None,
) -> list[tuple[Path, str]]:
    """
    Generate captions for a batch of images.
    
    Args:
        image_paths: List of image file paths
        model: Loaded Moondream2 model
        processor: Not used (kept for API compatibility)
        device: Device string
        lora_type: Type of LoRA being trained
        trigger_word: Optional trigger word to prepend
        progress_callback: Optional callback(current, total) for progress
        
    Returns:
        List of (image_path, caption) tuples
    """
    results = []
    total = len(image_paths)
    
    for i, image_path in enumerate(image_paths):
        try:
            caption = caption_image(
                image_path, model, processor, device, lora_type, trigger_word
            )
            results.append((image_path, caption))
        except Exception as e:
            print(f"Error captioning {image_path}: {e}")
            results.append((image_path, f"ERROR: {e}"))
        
        if progress_callback:
            progress_callback(i + 1, total)
    
    return results

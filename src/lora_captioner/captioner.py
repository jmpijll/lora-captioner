"""
Image captioning logic for LoRA Captioner.

Handles generating captions using Florence-2-PromptGen with
different strategies for Character, Style, and Concept LoRAs.
"""

from enum import Enum
from pathlib import Path
from typing import Literal

import torch
from PIL import Image


class LoRAType(str, Enum):
    """Types of LoRA training."""
    CHARACTER = "character"
    STYLE = "style"
    CONCEPT = "concept"


# Instruction prompts for each LoRA type
LORA_TYPE_INSTRUCTIONS = {
    LoRAType.CHARACTER: "<MORE_DETAILED_CAPTION>",
    LoRAType.STYLE: "<GENERATE_TAGS>",
    LoRAType.CONCEPT: "<DETAILED_CAPTION>",
}

# Alternative instructions for style LoRAs
STYLE_INSTRUCTIONS = {
    "tags": "<GENERATE_TAGS>",
    "natural": "<DETAILED_CAPTION>",
    "mixed": "<MIXED_CAPTION>",
}


def get_instruction(lora_type: LoRAType, style_mode: str = "tags") -> str:
    """
    Get the appropriate instruction prompt for a LoRA type.
    
    Args:
        lora_type: Type of LoRA being trained
        style_mode: For style LoRAs, which caption format to use
        
    Returns:
        Instruction prompt string
    """
    if lora_type == LoRAType.STYLE:
        return STYLE_INSTRUCTIONS.get(style_mode, STYLE_INSTRUCTIONS["tags"])
    return LORA_TYPE_INSTRUCTIONS[lora_type]


def caption_image(
    image_path: Path,
    model,
    processor,
    device: str,
    lora_type: LoRAType,
    trigger_word: str | None = None,
) -> str:
    """
    Generate a caption for a single image.
    
    Args:
        image_path: Path to the image file
        model: Loaded Florence-2 model
        processor: Loaded Florence-2 processor
        device: Device string (e.g., "cuda:0" or "cpu")
        lora_type: Type of LoRA being trained
        trigger_word: Optional trigger word to prepend
        
    Returns:
        Generated caption string
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Get instruction for this LoRA type
    instruction = get_instruction(lora_type)
    
    # Process inputs
    inputs = processor(
        text=instruction,
        images=image,
        return_tensors="pt"
    ).to(device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
        )
    
    # Decode output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process
    caption = processor.post_process_generation(
        generated_text,
        task=instruction,
        image_size=(image.width, image.height)
    )
    
    # Handle different output formats
    if isinstance(caption, dict):
        # Some instructions return dicts, extract the text
        caption = caption.get(instruction, str(caption))
    
    caption = str(caption).strip()
    
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
        model: Loaded Florence-2 model
        processor: Loaded Florence-2 processor
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

"""
Image captioning logic for LoRA Captioner.

Supports multiple models:
- BLIP: Stable, works with all transformers versions
- Florence-2: Better for LoRA training, requires transformers<=4.51.3
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


# BLIP prompts (conditional captioning)
BLIP_PROMPTS = {
    LoRAType.CHARACTER: "a photo of",
    LoRAType.STYLE: "an image of",
    LoRAType.CONCEPT: "a picture showing",
}

# Florence-2 instruction prompts
FLORENCE_INSTRUCTIONS = {
    LoRAType.CHARACTER: "<MORE_DETAILED_CAPTION>",
    LoRAType.STYLE: "<GENERATE_TAGS>",
    LoRAType.CONCEPT: "<DETAILED_CAPTION>",
}

# Qwen3-VL system prompt for ultra-detailed captions
QWEN_SYSTEM_PROMPT = """Write ONE ultra-detailed paragraph (10–16 sentences, ~180–320 words). Stay grounded in visible details. Include: subject micro-details (materials, textures, patterns, wear, reflections); people details if present (hair, skin tones, makeup, jewelry, fabric types, fit); environment depth (foreground/midground/background, signage/props, surface materials); lighting analysis (key/fill/back light, direction, softness, highlights, shadow shape); camera perspective (angle, lens feel, depth of field) and composition (leading lines, negative space, symmetry/asymmetry, visual hierarchy). No preface, no reasoning, no <think>."""

QWEN_USER_PROMPT = "Describe this image in detail."


def get_blip_prompt(lora_type: LoRAType) -> str:
    """Get BLIP conditional prompt for a LoRA type."""
    return BLIP_PROMPTS.get(lora_type, BLIP_PROMPTS[LoRAType.STYLE])


def get_florence_instruction(lora_type: LoRAType) -> str:
    """Get Florence-2 instruction for a LoRA type."""
    return FLORENCE_INSTRUCTIONS.get(lora_type, FLORENCE_INSTRUCTIONS[LoRAType.STYLE])


def caption_image(
    image_path: Path,
    model,
    processor,
    device: str,
    lora_type: LoRAType,
    trigger_word: str | None = None,
    model_type: str = "qwen",
) -> str:
    """
    Generate a caption for a single image.
    
    Args:
        image_path: Path to the image file
        model: Loaded model (Qwen3-VL, BLIP or Florence-2)
        processor: Model processor
        device: Device string (e.g., "cuda:0" or "cpu")
        lora_type: Type of LoRA being trained
        trigger_word: Optional trigger word to prepend
        model_type: Type of model ("qwen", "blip" or "florence")
        
    Returns:
        Generated caption string
    """
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    if model_type.lower() == "qwen":
        caption = _caption_with_qwen(image, model, processor, device, lora_type)
    elif model_type.lower() == "florence":
        caption = _caption_with_florence(image, model, processor, device, lora_type)
    else:
        caption = _caption_with_blip(image, model, processor, device, lora_type)
    
    # Prepend trigger word if specified
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    
    return caption


def _caption_with_blip(image: Image.Image, model, processor, device: str, lora_type: LoRAType) -> str:
    """Generate caption using BLIP model."""
    text_prompt = get_blip_prompt(lora_type)
    
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
    return caption.strip()


def _caption_with_florence(image: Image.Image, model, processor, device: str, lora_type: LoRAType) -> str:
    """Generate caption using Florence-2 model."""
    instruction = get_florence_instruction(lora_type)
    
    # Process inputs
    inputs = processor(text=instruction, images=image, return_tensors="pt")
    
    # Move to device with correct dtype
    model_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(device) for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=1,  # Greedy decoding for compatibility
            do_sample=False,
        )
    
    # Decode output
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Post-process using Florence-2's built-in parser
    caption = processor.post_process_generation(
        generated_text,
        task=instruction,
        image_size=(image.width, image.height)
    )
    
    # Handle different output formats from Florence-2
    if isinstance(caption, dict):
        caption = caption.get(instruction, "")
        if isinstance(caption, (dict, list)):
            caption = str(caption)
    
    return str(caption).strip()


def _caption_with_qwen(image: Image.Image, model, processor, device: str, lora_type: LoRAType) -> str:
    """Generate caption using Qwen3-VL model with ultra-detailed system prompt."""
    # Build messages with system prompt and image
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": QWEN_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": QWEN_USER_PROMPT},
            ],
        },
    ]
    
    # Apply chat template and process
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    # Move to device
    inputs = inputs.to(model.device)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )
    
    # Trim input tokens from output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # Decode output
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    
    return output_text[0].strip()


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

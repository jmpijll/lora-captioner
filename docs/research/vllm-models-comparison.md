# VLLM Models Comparison for LoRA Dataset Captioning

## Research Date: December 2024

This document compares Vision Language Models (VLLMs) suitable for creating captions for LoRA training datasets.

---

## Models Evaluated

### 1. Florence-2-large-PromptGen (MiaoshouAI) ⭐ RECOMMENDED

**Model ID:** `MiaoshouAI/Florence-2-large-PromptGen-v1.5`

**Why This Model:**
- Specifically fine-tuned for generating prompts/captions for diffusion model training
- Trained on Civitai images and cleaned tags - outputs match the format used in actual SD/Flux prompts
- No annoying "This image is about..." phrasing

**VRAM Requirements:** ~1-2 GB (extremely lightweight)

**Features:**
- `<GENERATE_TAGS>` - Danbooru-style tags
- `<CAPTION>` - One-line caption
- `<DETAILED_CAPTION>` - Structured caption with subject positions
- `<MORE_DETAILED_CAPTION>` - Very detailed description
- `<MIXED_CAPTION>` - Combined tags + description (ideal for Flux T5XXL + CLIP_L)

**Pros:**
- Minimal VRAM usage
- Lightning fast inference
- Purpose-built for LoRA training captions
- Multiple output modes

**Cons:**
- Less "creative" than larger models
- May miss nuanced artistic details

---

### 2. Moondream2

**Model ID:** `vikhyatk/moondream2`

**Architecture:** SigLIP (vision) + Phi-1.5 (language)

**Parameters:** 1.86 billion

**VRAM Requirements:** ~4-6 GB

**Pros:**
- Small and efficient
- Good general-purpose captioning
- Active development

**Cons:**
- Not specifically tuned for LoRA training
- May produce generic captions

---

### 3. JoyCaption (fancyfeast)

**Model ID:** `fancyfeast/llama-joycaption-beta-one-hf-llava`

**Architecture:** LLaVA-based with Llama 3.1

**VRAM Requirements:** ~17 GB native (can quantize to 8-bit/4-bit)

**Key Features:**
- Built specifically for diffusion model training captions
- Uncensored - equal SFW/NSFW coverage
- Multiple captioning modes:
  - Descriptive Caption
  - Straightforward Caption
  - Stable Diffusion Prompt mode
  - MidJourney Prompt mode
  - Danbooru tag list

**Pros:**
- Highest quality captions
- Purpose-built for LoRA training
- Very flexible output styles

**Cons:**
- High VRAM requirement
- Slower inference
- Requires quantization for most consumer GPUs

---

### 4. Microsoft Florence-2-base/large

**Model ID:** `microsoft/Florence-2-base` or `microsoft/Florence-2-large`

**VRAM Requirements:** ~2-4 GB

**Tasks Supported:**
- `<CAPTION>` - Brief caption
- `<DETAILED_CAPTION>` - Detailed description
- `<MORE_DETAILED_CAPTION>` - Extended description
- Also: OCR, object detection, segmentation

**Pros:**
- Very efficient
- Well-documented
- Multiple vision tasks

**Cons:**
- Generic captions, not LoRA-optimized
- May produce "This image shows..." style output

---

## Recommendation Matrix

| Use Case | Primary Model | Fallback |
|----------|--------------|----------|
| Low VRAM (<4GB) | Florence-2-PromptGen | Florence-2-base |
| Medium VRAM (4-8GB) | Florence-2-PromptGen | Moondream2 |
| High VRAM (12GB+) | JoyCaption (quantized) | Florence-2-PromptGen |
| CPU-only | Florence-2-PromptGen | Florence-2-base |

---

## Selected Model: Florence-2-large-PromptGen-v1.5

**Rationale:**
1. Purpose-built for LoRA training dataset creation
2. Minimal resource requirements (~1-2GB VRAM)
3. Fast inference suitable for batch processing
4. Multiple output modes for different LoRA types
5. Can run on CPU with acceptable performance
6. Outputs in prompt-ready format (no post-processing needed)

---

## Sources

- https://huggingface.co/MiaoshouAI/Florence-2-large-PromptGen-v1.5
- https://huggingface.co/microsoft/Florence-2-base
- https://github.com/fpgaminer/joycaption
- https://huggingface.co/vikhyatk/moondream2
- https://datasciencedojo.com/blog/vision-language-models-moondream-2/

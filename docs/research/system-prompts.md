# System Prompts for LoRA Dataset Captioning

## Research Date: December 2024

This document contains optimized prompts for the Florence-2-PromptGen model for different LoRA training scenarios.

---

## Model: Florence-2-large-PromptGen-v1.5

### Available Instruction Prompts

| Instruction | Output Type |
|-------------|-------------|
| `<GENERATE_TAGS>` | Danbooru-style tags |
| `<CAPTION>` | One-line caption |
| `<DETAILED_CAPTION>` | Structured caption with positions |
| `<MORE_DETAILED_CAPTION>` | Very detailed description |
| `<MIXED_CAPTION>` | Tags + description (for Flux) |

---

## Recommended Prompts by LoRA Type

### Character LoRA

**Instruction:** `<MORE_DETAILED_CAPTION>`

**Post-processing:** Prepend trigger word, optionally remove character-specific descriptions

**Why:** Detailed captions capture pose, expression, environment while we manually handle trigger words.

---

### Style LoRA

**Option 1: Tag-based (simpler)**
- **Instruction:** `<GENERATE_TAGS>`
- Add style trigger word as first tag

**Option 2: Natural language (more flexible)**
- **Instruction:** `<DETAILED_CAPTION>`
- Prepend style trigger word

---

### Concept LoRA

**Instruction:** `<DETAILED_CAPTION>`

**Why:** Structured format captures spatial relationships and context well.

---

## Custom System Prompts (for models that support it)

### Character LoRA System Prompt
```
You are an image captioner for AI training datasets. Describe the image focusing on:
- Pose and body position
- Facial expression and gaze direction
- Background environment and setting
- Lighting conditions and mood
- Actions being performed
- Camera angle and framing

Do NOT describe:
- The subject's permanent physical features (hair color, eye color, body type)
- Signature clothing or accessories that appear in every image
- The subject's identity or name

Write in natural language, suitable for text-to-image prompts. Be specific and use visual descriptors.
```

### Style LoRA System Prompt
```
You are an image captioner for AI training datasets. Describe what is depicted in the image:
- Main subjects and their arrangement
- Scene and setting
- Objects and their relationships
- Composition and framing
- Color palette (briefly)

Focus on WHAT is shown, not HOW it is rendered artistically.
Write in natural language suitable for text-to-image generation prompts.
```

### Concept LoRA System Prompt
```
You are an image captioner for AI training datasets. Describe the image focusing on:
- The scene and environment
- How objects and elements are positioned
- Context and setting
- Actions or states depicted
- Relationships between elements

Write in natural language. Be descriptive but concise.
```

---

## Florence-2-PromptGen Usage Notes

1. **No custom system prompts** - Use instruction tokens only
2. **Instruction tokens are task selectors** - Just pass the token, model handles the rest
3. **Post-process for trigger words** - Prepend manually after generation
4. **Batch processing friendly** - Same instruction for all images of same LoRA type

---

## Example Implementation

```python
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image

model = AutoModelForCausalLM.from_pretrained(
    "MiaoshouAI/Florence-2-large-PromptGen-v1.5", 
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    "MiaoshouAI/Florence-2-large-PromptGen-v1.5", 
    trust_remote_code=True
)

def caption_image(image_path, instruction="<MORE_DETAILED_CAPTION>", trigger_word=None):
    image = Image.open(image_path)
    inputs = processor(text=instruction, images=image, return_tensors="pt")
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3
    )
    
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    caption = processor.post_process_generation(
        generated_text, 
        task=instruction, 
        image_size=(image.width, image.height)
    )
    
    # Add trigger word if specified
    if trigger_word:
        caption = f"{trigger_word}, {caption}"
    
    return caption
```

---

## Output Format Examples

### `<GENERATE_TAGS>` Output
```
1girl, solo, standing, long_hair, brown_hair, dress, outdoors, sky, clouds, looking_at_viewer
```

### `<CAPTION>` Output
```
A young woman with long brown hair stands outdoors against a cloudy sky
```

### `<DETAILED_CAPTION>` Output
```
In the center of the image, a young woman stands facing the viewer. She has long brown hair and wears a blue dress. The background shows a cloudy sky with scattered white clouds.
```

### `<MORE_DETAILED_CAPTION>` Output
```
The image features a young woman positioned centrally, standing upright and facing directly toward the viewer. Her long, flowing brown hair cascades past her shoulders. She is dressed in an elegant blue dress with subtle detailing. The background presents a dramatic sky filled with billowing white and gray clouds, suggesting an outdoor setting during late afternoon. The lighting is soft and diffused, creating gentle shadows on her features.
```

### `<MIXED_CAPTION>` Output
```
A young woman with long brown hair stands outdoors in a blue dress, looking at the viewer. Tags: 1girl, solo, standing, long_hair, brown_hair, dress, outdoors, sky, clouds
```

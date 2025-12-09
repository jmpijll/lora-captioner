# Captioning Strategies for LoRA Training

## Research Date: December 2024

This document outlines best practices for captioning images for different types of LoRA training.

---

## Core Principle

**Caption what you want to be VARIABLE, leave out what you want to be FIXED.**

When training a LoRA, the model learns to associate:
- Captioned elements → Things that can change with prompting
- Uncaptioned elements → The "essence" of what the LoRA represents

---

## LoRA Type: Character

### Goal
Train the model to recognize a specific character's appearance while allowing variation in poses, expressions, backgrounds, and situations.

### Captioning Strategy

**DO Caption:**
- Pose and body position
- Facial expression
- Background/environment
- Lighting and mood
- Actions being performed
- Camera angle/framing
- Any temporary elements (accessories that change between images)

**DO NOT Caption:**
- Core character features (hair color, eye color, defining traits)
- Signature clothing/outfit
- Distinguishing accessories (always-present items)
- Character name (use trigger word instead)

### Example Caption Format
```
[trigger_word] gazes upward with a look of wonder, her eyes wide and lips 
slightly parted as if captivated by something above. Warm, glowing lights 
in the background add a soft, ambient glow to the scene, highlighting her features
```

### Trigger Word
- Use a unique, non-dictionary word (e.g., `sks`, `ohwx`, `jm1char`)
- Place at the beginning of every caption
- Don't use the character's actual name (causes conflicts with existing model knowledge)

---

## LoRA Type: Style

### Goal
Train the model to replicate an artistic style across any subject matter.

### Captioning Strategy

**Option A: No Captions (Simpler)**
- Use only a trigger word (e.g., `artstyle_moebius`)
- Best when you want to capture EVERYTHING about the visual style
- Requires diverse training images (different subjects, compositions)
- Risk: May overfit to specific subjects if dataset is homogeneous

**Option B: Full Natural Language Captions (More Flexible)**
- Caption everything in detail
- Results in better prompt adherence
- LoRA can apply style to subjects outside training data more easily
- More work but more versatile results

### Recommended Approach
1. Ensure diverse training data (varied subjects, compositions)
2. Use trigger word + minimal style description
3. Caption subject matter to avoid overfitting

### Example
```
[style_trigger] a woman sitting in a cafe, warm lighting, coffee cup on table
```

---

## LoRA Type: Concept

### Goal
Train the model to understand a specific object, clothing item, pose, or abstract concept.

### Captioning Strategy

**DO Caption:**
- Context and environment
- Other elements in the scene
- How the concept is being used/worn/positioned
- Variations (colors, sizes if they vary)

**DO NOT Caption:**
- The core concept itself
- Defining features of the concept

### Example (Training a specific jacket design)
```
[jacket_trigger] worn by a young man standing in an urban street, 
hands in pockets, evening lighting, graffiti wall behind
```

---

## Caption Format Recommendations

### For Flux Models
- Natural language descriptions work best
- T5XXL encoder handles longer, more detailed text
- Can use mixed format (description + tags)

### For SDXL/SD1.5
- Shorter, tag-like captions often work better
- Booru-style tags are effective
- Trigger word at the start

---

## Practical Tips

### 1. Consistency
- Use the same trigger word format throughout
- Maintain consistent caption style (all natural language OR all tags)

### 2. Caption Quality
- Be specific and descriptive
- Avoid vague terms like "nice", "beautiful"
- Use concrete visual descriptors

### 3. What NOT to Include
- "This image shows..."
- "A photograph of..."
- Image quality descriptors (unless training quality-aware LoRA)
- Speculation about what isn't visible

### 4. Dataset Size Guidelines
- Character LoRA: 15-50 high-quality images
- Style LoRA: 30-100 diverse images
- Concept LoRA: 20-40 varied examples

---

## VLLM Prompting for Each Type

### Character LoRA Prompt
```
Describe this image in detail, focusing on the pose, expression, 
background, lighting, and composition. Do not describe the main 
character's permanent physical features like hair color, eye color, 
or their signature clothing.
```

### Style LoRA Prompt
```
Describe the subject and composition of this image. Focus on what 
is depicted, the scene, objects, and their arrangement. Describe 
in natural language suitable for image generation prompts.
```

### Concept LoRA Prompt
```
Describe this image focusing on the context, environment, and how 
elements interact. Describe the scene and setting in detail. Use 
natural language suitable for training data.
```

---

## Sources

- https://civitai.com/articles/8487/understanding-prompting-and-captioning-for-loras
- https://civitai.com/articles/7097/flux-complete-lora-settings-and-dataset-guide
- https://www.reddit.com/r/StableDiffusion/comments/1b3ygtz/lora_training_how_what_to_caption/
- https://www.reddit.com/r/FluxAI/comments/1fcrj3j/what_exactly_to_caption_for_flux_lora_training/

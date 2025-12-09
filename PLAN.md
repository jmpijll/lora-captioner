# LoRA Captioner - Development Plan

## Project Overview

A command-line tool to caption images for LoRA training datasets using a lightweight Vision Language Model (VLLM).

---

## Phase 1: Project Setup & Foundation

### Step 1.1: Repository Setup
- [x] Initialize git repository
- [ ] Create GitHub repository at github.com/jmpijll/lora-captioner
- [ ] Set up .gitignore for Python
- [ ] Create initial README.md
- [ ] Add LICENSE (MIT recommended)

### Step 1.2: Python Project Structure
- [ ] Create pyproject.toml with dependencies
- [ ] Set up src/lora_captioner package structure
- [ ] Create requirements.txt for pip users
- [ ] Add dev dependencies (pytest, black, ruff)

### Step 1.3: Documentation Foundation
- [x] Create docs/research folder with research findings
- [x] Document VLLM model selection rationale
- [x] Document captioning strategies for Character/Style/Concept
- [x] Document system prompts and usage

---

## Phase 2: Core Implementation

### Step 2.1: CLI Framework
- [ ] Set up Click/Typer CLI framework
- [ ] Implement basic argument parsing:
  - `--input` / `-i`: Input folder path
  - `--dataset-name` / `-n`: Dataset name for renaming
  - `--lora-type` / `-t`: character | style | concept
  - `--device`: cuda | cpu | auto (default: auto)
  - `--trigger-word` / `-w`: Optional trigger word to prepend

### Step 2.2: Model Management
- [ ] Implement model download/cache logic
- [ ] Auto-download Florence-2-PromptGen on first run
- [ ] Store model in `~/.cache/lora-captioner/` or user-specified path
- [ ] Implement model existence check
- [ ] Add `--model-path` override option

### Step 2.3: Device Detection
- [ ] Detect available hardware (CUDA, CPU)
- [ ] Check VRAM availability if GPU present
- [ ] Auto-select device based on model requirements
- [ ] Implement `--device` override flag
- [ ] Add `--force-cpu` convenience flag

---

## Phase 3: Image Processing Pipeline

### Step 3.1: Image Discovery
- [ ] Scan input folder for supported formats (jpg, jpeg, png, webp, bmp)
- [ ] Handle nested folders (optional `--recursive` flag)
- [ ] Sort images for consistent ordering
- [ ] Report image count before processing

### Step 3.2: Image Renaming
- [ ] Implement rename logic: `{dataset_name}_{number:04d}.{ext}`
- [ ] Preserve original file extensions
- [ ] Handle naming conflicts
- [ ] Create mapping log for traceability
- [ ] Add `--no-rename` flag to skip renaming

### Step 3.3: Caption Generation
- [ ] Load Florence-2-PromptGen model
- [ ] Select instruction based on LoRA type:
  - Character: `<MORE_DETAILED_CAPTION>`
  - Style: `<GENERATE_TAGS>` or `<DETAILED_CAPTION>`
  - Concept: `<DETAILED_CAPTION>`
- [ ] Process images in batches (configurable batch size)
- [ ] Add trigger word prefix if specified
- [ ] Display progress bar

### Step 3.4: Caption File Writing
- [ ] Write .txt file with same base name as image
- [ ] UTF-8 encoding
- [ ] Handle write errors gracefully
- [ ] Verify file creation

---

## Phase 4: Advanced Features

### Step 4.1: Configuration
- [ ] Add `--config` flag for JSON/YAML config files
- [ ] Support for per-project settings
- [ ] Default configuration template generation

### Step 4.2: Caption Customization
- [ ] Add `--instruction` flag for custom Florence-2 instruction
- [ ] Add `--max-length` for caption length control
- [ ] Add `--caption-style` (tags | natural | mixed)

### Step 4.3: Output Options
- [ ] Add `--output` / `-o` for custom output directory
- [ ] Add `--dry-run` to preview without writing
- [ ] Add `--overwrite` flag (default: skip existing)
- [ ] Generate summary report

---

## Phase 5: Quality & Polish

### Step 5.1: Error Handling
- [ ] Graceful handling of corrupt images
- [ ] Network error recovery for model download
- [ ] Clear error messages with actionable guidance
- [ ] Logging to file (optional)

### Step 5.2: Testing
- [ ] Unit tests for core functions
- [ ] Integration tests with sample images
- [ ] Test on Windows/Linux/macOS
- [ ] Test CPU-only mode

### Step 5.3: Documentation
- [ ] Complete README with examples
- [ ] Add CONTRIBUTING.md
- [ ] Add CHANGELOG.md
- [ ] Add usage examples for each LoRA type
- [ ] Document all CLI options

### Step 5.4: Distribution
- [ ] Publish to PyPI
- [ ] Create GitHub releases
- [ ] Add installation instructions for various methods

---

## Technical Specifications

### Dependencies
```
transformers>=4.36.0
torch>=2.0.0
Pillow>=10.0.0
click>=8.0.0 (or typer>=0.9.0)
tqdm>=4.65.0
huggingface-hub>=0.20.0
```

### Minimum Requirements
- Python 3.10+
- 4GB RAM (CPU mode)
- 2GB VRAM (GPU mode with Florence-2-PromptGen)

### Model Details
- **Primary Model:** MiaoshouAI/Florence-2-large-PromptGen-v1.5
- **Model Size:** ~1.5GB download
- **VRAM Usage:** ~1-2GB
- **Inference Speed:** ~0.5-2 sec/image (GPU), ~2-5 sec/image (CPU)

---

## CLI Usage Examples

```bash
# Basic usage - Character LoRA
lora-captioner -i ./my_images -n "my_character" -t character -w "sks_person"

# Style LoRA with tags
lora-captioner -i ./style_images -n "moebius_style" -t style

# Concept LoRA, force CPU
lora-captioner -i ./concept_images -n "special_hat" -t concept --device cpu

# Custom output location
lora-captioner -i ./raw -n "dataset" -t character -o ./processed

# Dry run to preview
lora-captioner -i ./images -n "test" -t style --dry-run
```

---

## File Structure

```
lora-captioner/
├── src/
│   └── lora_captioner/
│       ├── __init__.py
│       ├── __main__.py
│       ├── cli.py
│       ├── captioner.py
│       ├── model_manager.py
│       ├── image_processor.py
│       └── utils.py
├── tests/
│   ├── __init__.py
│   ├── test_captioner.py
│   └── test_image_processor.py
├── docs/
│   └── research/
│       ├── vllm-models-comparison.md
│       ├── captioning-strategies.md
│       └── system-prompts.md
├── AGENTS.md
├── PLAN.md
├── README.md
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── .gitignore
```

---

## Milestones

| Milestone | Description | Target |
|-----------|-------------|--------|
| M1 | Basic CLI + Model Download | Phase 1-2 |
| M2 | Image Processing + Captioning | Phase 3 |
| M3 | Full Feature Set | Phase 4 |
| M4 | Release Ready | Phase 5 |

---

## Next Steps

1. Create GitHub repository
2. Set up project structure with pyproject.toml
3. Implement CLI skeleton
4. Implement model download logic
5. Implement core captioning pipeline

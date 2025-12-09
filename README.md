# LoRA Captioner

A command-line tool to automatically caption images for LoRA training datasets using Vision Language Models (VLLMs).

## Features

- **Automatic Image Captioning**: Uses BLIP (Salesforce/blip-image-captioning-large), a robust model for generating descriptive captions
- **Multiple LoRA Types**: Optimized captioning strategies for Character, Style, and Concept LoRAs
- **Low Resource Usage**: Runs on as little as 2GB VRAM (GPU) or 4GB RAM (CPU)
- **Automatic Model Download**: Downloads the VLLM model on first run
- **Smart Device Detection**: Automatically selects GPU or CPU based on available hardware
- **Batch Renaming**: Renames images to consistent dataset format

## Installation

### Prerequisites

1. **Python 3.10 or higher** - [Download from python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"

2. **PyTorch** (recommended for GPU acceleration):
   ```bash
   # For CUDA 11.8
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   
   # For CPU only
   pip install torch
   ```

### Install from PyPI (coming soon)

```bash
pip install lora-captioner
```

### Install from Source

```bash
git clone https://github.com/jmpijll/lora-captioner.git
cd lora-captioner
pip install -e .
```

### Verify Installation

```bash
lora-captioner --help
```

## Quick Start

```bash
# Caption images for a character LoRA
lora-captioner -i ./my_images -n "my_character" -t character -w "sks_person"

# Caption images for a style LoRA
lora-captioner -i ./style_images -n "moebius_style" -t style

# Caption images for a concept LoRA
lora-captioner -i ./concept_images -n "special_hat" -t concept
```

## Usage

```
lora-captioner [OPTIONS]

Options:
  -i, --input PATH          Input folder containing images [required]
  -n, --dataset-name TEXT   Name for the dataset (used in file renaming) [required]
  -t, --lora-type TYPE      Type of LoRA: character, style, or concept [required]
  -w, --trigger-word TEXT   Trigger word to prepend to captions
  -o, --output PATH         Output folder (default: same as input)
  --device TEXT             Device to use: auto, cuda, or cpu (default: auto)
  --no-rename               Skip renaming images
  --recursive               Search for images in subdirectories
  --dry-run                 Preview actions without making changes
  --version                 Show version and exit
  --help                    Show this message and exit
```

## LoRA Type Strategies

### Character LoRA
- Captions describe pose, expression, background, lighting
- Does NOT describe permanent character features (hair color, signature outfit)
- Use trigger word to identify the character

### Style LoRA  
- Can use tag-based or natural language captions
- Best with diverse training images
- Trigger word identifies the style

### Concept LoRA
- Captions describe context and environment
- Does NOT describe the core concept being trained
- Trigger word identifies the concept

## Requirements

- Python 3.10+
- 4GB RAM minimum (CPU mode)
- 2GB VRAM minimum (GPU mode)

## Model Information

This tool uses [Salesforce/blip-image-captioning-large](https://huggingface.co/Salesforce/blip-image-captioning-large), a robust vision-language model for generating descriptive image captions. BLIP was chosen for its stability and broad compatibility.

## Documentation

- [VLLM Model Comparison](docs/research/vllm-models-comparison.md)
- [Captioning Strategies](docs/research/captioning-strategies.md)
- [System Prompts Reference](docs/research/system-prompts.md)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read the [AGENTS.md](AGENTS.md) file for development guidelines.

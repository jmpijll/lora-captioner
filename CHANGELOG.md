# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-02-01

### Added
- **Qwen3-VL-2B-Instruct** as the new default captioning model
- Ultra-detailed caption generation with comprehensive descriptions including:
  - Subject micro-details (materials, textures, patterns, wear, reflections)
  - People details (hair, skin tones, makeup, jewelry, fabric types, fit)
  - Environment depth (foreground/midground/background, signage, props)
  - Lighting analysis (key/fill/back light, direction, softness, shadows)
  - Camera perspective and composition (angle, lens feel, depth of field)
- Model selection via `--model` flag: `qwen` (default), `florence`, or `blip`

### Changed
- Default model changed from Florence-2 to Qwen3-VL-2B-Instruct
- Updated `transformers` requirement to `>=4.45.0` for Qwen3-VL support
- Added `qwen-vl-utils` and `accelerate` dependencies

### Fixed
- Improved error messages with model-specific troubleshooting tips

## [0.1.0] - 2024-12-10

### Added
- Initial release
- Command-line interface with Click framework
- Support for three LoRA types: Character, Style, and Concept
- Automatic image discovery (jpg, jpeg, png, webp, bmp, gif, tiff)
- Image renaming to dataset format (`{name}_{number:04d}.{ext}`)
- Caption generation using Florence-2-large-PromptGen-v1.5
- Automatic model download on first run
- GPU/CPU device auto-detection with manual override
- Trigger word prepending to captions
- Progress bar with tqdm
- Dry-run mode for previewing changes
- Recursive directory search option
- Rename log generation for traceability

### Documentation
- Research documentation on VLLM model selection
- Captioning strategies for Character, Style, and Concept LoRAs
- System prompts reference
- Development plan (PLAN.md)
- Agent guidelines (AGENTS.md)

[Unreleased]: https://github.com/jmpijll/lora-captioner/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/jmpijll/lora-captioner/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jmpijll/lora-captioner/releases/tag/v0.1.0

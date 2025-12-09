# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/jmpijll/lora-captioner/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jmpijll/lora-captioner/releases/tag/v0.1.0

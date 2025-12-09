"""
Image file discovery and processing for LoRA Captioner.

Handles finding images, renaming them, and writing caption files.
"""

from pathlib import Path
from typing import Iterator

# Supported image formats
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}


def discover_images(
    input_dir: Path,
    recursive: bool = False,
) -> list[Path]:
    """
    Find all supported image files in a directory.
    
    Args:
        input_dir: Directory to search
        recursive: Whether to search subdirectories
        
    Returns:
        Sorted list of image file paths
    """
    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"
    
    images = []
    for path in input_dir.glob(pattern):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)
    
    # Sort for consistent ordering
    images.sort(key=lambda p: p.name.lower())
    
    return images


def generate_new_names(
    image_paths: list[Path],
    dataset_name: str,
    output_dir: Path | None = None,
) -> list[tuple[Path, Path]]:
    """
    Generate new file names for images.
    
    Args:
        image_paths: Original image paths
        dataset_name: Base name for the dataset
        output_dir: Output directory (default: same as original)
        
    Returns:
        List of (original_path, new_path) tuples
    """
    mappings = []
    
    for i, original_path in enumerate(image_paths, start=1):
        extension = original_path.suffix.lower()
        new_name = f"{dataset_name}_{i:04d}{extension}"
        
        if output_dir:
            new_path = output_dir / new_name
        else:
            new_path = original_path.parent / new_name
        
        mappings.append((original_path, new_path))
    
    return mappings


def rename_images(
    mappings: list[tuple[Path, Path]],
    dry_run: bool = False,
) -> list[Path]:
    """
    Rename images according to the mappings.
    
    Args:
        mappings: List of (original_path, new_path) tuples
        dry_run: If True, don't actually rename files
        
    Returns:
        List of new file paths
    """
    new_paths = []
    
    for original, new in mappings:
        if dry_run:
            print(f"Would rename: {original.name} -> {new.name}")
        else:
            # Handle case where new path might already exist
            if new.exists() and new != original:
                raise FileExistsError(f"Target file already exists: {new}")
            
            original.rename(new)
        
        new_paths.append(new)
    
    return new_paths


def write_caption_file(
    image_path: Path,
    caption: str,
    dry_run: bool = False,
) -> Path:
    """
    Write a caption file for an image.
    
    Caption file will have the same name as the image with .txt extension.
    
    Args:
        image_path: Path to the image file
        caption: Caption text to write
        dry_run: If True, don't actually write file
        
    Returns:
        Path to the caption file
    """
    caption_path = image_path.with_suffix(".txt")
    
    if dry_run:
        print(f"Would write caption to: {caption_path.name}")
        print(f"  Caption: {caption[:100]}{'...' if len(caption) > 100 else ''}")
    else:
        caption_path.write_text(caption, encoding="utf-8")
    
    return caption_path


def write_all_captions(
    captions: list[tuple[Path, str]],
    dry_run: bool = False,
) -> list[Path]:
    """
    Write caption files for all images.
    
    Args:
        captions: List of (image_path, caption) tuples
        dry_run: If True, don't actually write files
        
    Returns:
        List of caption file paths
    """
    caption_paths = []
    
    for image_path, caption in captions:
        caption_path = write_caption_file(image_path, caption, dry_run)
        caption_paths.append(caption_path)
    
    return caption_paths


def create_rename_log(
    mappings: list[tuple[Path, Path]],
    output_dir: Path,
    dry_run: bool = False,
) -> Path | None:
    """
    Create a log file documenting the rename mappings.
    
    Args:
        mappings: List of (original_path, new_path) tuples
        output_dir: Directory to write the log file
        dry_run: If True, don't actually write file
        
    Returns:
        Path to the log file, or None if dry_run
    """
    log_path = output_dir / "rename_log.txt"
    
    if dry_run:
        print(f"Would write rename log to: {log_path}")
        return None
    
    lines = ["# Original Name -> New Name\n"]
    for original, new in mappings:
        lines.append(f"{original.name} -> {new.name}\n")
    
    log_path.write_text("".join(lines), encoding="utf-8")
    
    return log_path

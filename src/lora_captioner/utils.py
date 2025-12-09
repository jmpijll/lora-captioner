"""
Utility functions for LoRA Captioner.
"""

import sys
from pathlib import Path


def get_vram_info() -> dict | None:
    """
    Get information about available GPU VRAM.
    
    Returns:
        Dict with VRAM info, or None if no GPU available
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return None
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        
        total_vram = props.total_memory / (1024**3)  # GB
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        free = total_vram - reserved
        
        return {
            "device_name": props.name,
            "total_gb": round(total_vram, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
        }
    except Exception:
        return None


def get_system_info() -> dict:
    """
    Get basic system information.
    
    Returns:
        Dict with system info
    """
    import platform
    
    info = {
        "python_version": sys.version,
        "platform": platform.system(),
        "platform_release": platform.release(),
        "architecture": platform.machine(),
    }
    
    # Add GPU info if available
    vram_info = get_vram_info()
    if vram_info:
        info["gpu"] = vram_info
    else:
        info["gpu"] = None
    
    return info


def format_file_size(size_bytes: int) -> str:
    """
    Format a file size in bytes to a human-readable string.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def count_images(directory: Path) -> int:
    """
    Count the number of supported image files in a directory.
    
    Args:
        directory: Directory to count images in
        
    Returns:
        Number of images found
    """
    from lora_captioner.image_processor import SUPPORTED_EXTENSIONS
    
    count = 0
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            count += 1
    return count

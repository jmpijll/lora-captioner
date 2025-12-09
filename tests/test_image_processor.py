"""
Tests for image processing functionality.
"""

import pytest
from pathlib import Path

from lora_captioner.image_processor import (
    SUPPORTED_EXTENSIONS,
    discover_images,
    generate_new_names,
)


def test_supported_extensions():
    """Test that common image formats are supported."""
    assert ".jpg" in SUPPORTED_EXTENSIONS
    assert ".jpeg" in SUPPORTED_EXTENSIONS
    assert ".png" in SUPPORTED_EXTENSIONS
    assert ".webp" in SUPPORTED_EXTENSIONS


def test_generate_new_names():
    """Test filename generation."""
    paths = [
        Path("/test/image1.jpg"),
        Path("/test/image2.png"),
        Path("/test/image3.webp"),
    ]
    
    mappings = generate_new_names(paths, "my_dataset")
    
    assert len(mappings) == 3
    assert mappings[0][1].name == "my_dataset_0001.jpg"
    assert mappings[1][1].name == "my_dataset_0002.png"
    assert mappings[2][1].name == "my_dataset_0003.webp"


def test_generate_new_names_with_output_dir():
    """Test filename generation with custom output directory."""
    paths = [Path("/test/image1.jpg")]
    output_dir = Path("/output")
    
    mappings = generate_new_names(paths, "dataset", output_dir)
    
    assert mappings[0][1] == Path("/output/dataset_0001.jpg")

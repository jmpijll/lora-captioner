"""
Command-line interface for LoRA Captioner.
"""

import sys
from pathlib import Path

import click
from tqdm import tqdm

from lora_captioner import __version__
from lora_captioner.captioner import LoRAType, caption_image
from lora_captioner.image_processor import (
    discover_images,
    generate_new_names,
    rename_images,
    write_caption_file,
    create_rename_log,
)
from lora_captioner.model_manager import load_model


@click.command()
@click.option(
    "-i", "--input",
    "input_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Input folder containing images"
)
@click.option(
    "-n", "--dataset-name",
    required=True,
    help="Name for the dataset (used in file renaming)"
)
@click.option(
    "-t", "--lora-type",
    type=click.Choice(["character", "style", "concept"], case_sensitive=False),
    required=True,
    help="Type of LoRA being trained"
)
@click.option(
    "-w", "--trigger-word",
    default=None,
    help="Trigger word to prepend to captions"
)
@click.option(
    "-o", "--output",
    "output_path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Output folder (default: same as input)"
)
@click.option(
    "--device",
    type=click.Choice(["auto", "cuda", "cpu"], case_sensitive=False),
    default="auto",
    help="Device to use for inference"
)
@click.option(
    "--no-rename",
    is_flag=True,
    default=False,
    help="Skip renaming images"
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Preview actions without making changes"
)
@click.option(
    "--recursive",
    is_flag=True,
    default=False,
    help="Search for images in subdirectories"
)
@click.option(
    "--model",
    type=click.Choice(["florence", "blip"], case_sensitive=False),
    default="florence",
    help="Model to use: florence (default, better for LoRA) or blip (fallback)"
)
@click.version_option(version=__version__, prog_name="lora-captioner")
def main(
    input_path: Path,
    dataset_name: str,
    lora_type: str,
    trigger_word: str | None,
    output_path: Path | None,
    device: str,
    no_rename: bool,
    dry_run: bool,
    recursive: bool,
    model: str,
):
    """
    Caption images for LoRA training datasets.
    
    Uses Florence-2-PromptGen to generate training-ready captions
    optimized for different LoRA types (character, style, concept).
    """
    click.echo(f"{'='*50}")
    click.echo(f"LoRA Captioner v{__version__}")
    click.echo(f"{'='*50}")
    click.echo(f"Input folder:  {input_path}")
    click.echo(f"Dataset name:  {dataset_name}")
    click.echo(f"LoRA type:     {lora_type}")
    click.echo(f"Trigger word:  {trigger_word or '(none)'}")
    click.echo(f"Device:        {device}")
    click.echo(f"Model:         {model}")
    click.echo(f"Rename files:  {'No' if no_rename else 'Yes'}")
    
    if dry_run:
        click.echo("\n[!] DRY RUN MODE - No changes will be made\n")
    
    # Set output directory
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)
        if not dry_run:
            output_path.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Discover images
    click.echo("\n[1/4] Discovering images...")
    images = discover_images(input_path, recursive=recursive)
    
    if not images:
        click.echo("ERROR: No supported images found in the input folder.")
        click.echo("   Supported formats: jpg, jpeg, png, webp, bmp, gif, tiff")
        sys.exit(1)
    
    click.echo(f"   Found {len(images)} images")
    
    # Step 2: Rename images (if enabled)
    if not no_rename:
        click.echo("\n[2/4] Renaming images...")
        mappings = generate_new_names(images, dataset_name, output_path)
        
        if dry_run:
            for original, new in mappings[:5]:  # Show first 5
                click.echo(f"   {original.name} -> {new.name}")
            if len(mappings) > 5:
                click.echo(f"   ... and {len(mappings) - 5} more")
        else:
            try:
                images = rename_images(mappings, dry_run=False)
                create_rename_log(mappings, output_path, dry_run=False)
                click.echo(f"   Renamed {len(images)} images")
            except FileExistsError as e:
                click.echo(f"ERROR: {e}")
                sys.exit(1)
    else:
        click.echo("\n[2/4] Skipping rename (--no-rename)")
        # If output differs from input, we'd need to copy files
        # For now, just work with original paths
    
    # Step 3: Load model
    if not dry_run:
        click.echo("\n[3/4] Loading captioning model...")
        if model == "florence":
            click.echo("   Model: microsoft/Florence-2-large")
        else:
            click.echo("   Model: Salesforce/blip-image-captioning-large (fallback)")
        click.echo("   (This may take a moment on first run as the model downloads)")
        
        try:
            loaded_model, processor, device_str = load_model(device=device, model_type=model)
            click.echo(f"   Model loaded on {device_str}")
        except Exception as e:
            click.echo(f"ERROR: Failed to load model: {e}")
            if model == "florence":
                click.echo("   TIP: Try reinstalling with: pip install -e .")
                click.echo("   Or use --model blip as fallback")
            sys.exit(1)
    else:
        click.echo("\n[3/4] Would load captioning model (dry run)")
        loaded_model, processor, device_str = None, None, "cpu"
    
    # Step 4: Generate captions
    click.echo("\n[4/4] Generating captions...")
    lora_type_enum = LoRAType(lora_type.lower())
    
    captions_generated = 0
    errors = []
    
    # Use tqdm for progress
    for image_path in tqdm(images, desc="Captioning", unit="img"):
        if dry_run:
            caption = f"[DRY RUN] Caption for {image_path.name}"
            if trigger_word:
                caption = f"{trigger_word}, {caption}"
        else:
            try:
                caption = caption_image(
                    image_path=image_path,
                    model=loaded_model,
                    processor=processor,
                    device=device_str,
                    lora_type=lora_type_enum,
                    trigger_word=trigger_word,
                    model_type=model,
                )
            except Exception as e:
                errors.append((image_path, str(e)))
                continue
        
        # Write caption file
        caption_path = image_path.with_suffix(".txt")
        if dry_run:
            pass  # Don't write in dry run
        else:
            try:
                caption_path.write_text(caption, encoding="utf-8")
                captions_generated += 1
            except Exception as e:
                errors.append((image_path, f"Write error: {e}"))
    
    # Step 5: Summary
    click.echo(f"\n{'='*50}")
    click.echo("SUMMARY")
    click.echo(f"{'='*50}")
    click.echo(f"   Images processed: {len(images)}")
    click.echo(f"   Captions created: {captions_generated if not dry_run else '(dry run)'}")
    
    if errors:
        click.echo(f"\nWARNING: Errors ({len(errors)}):")
        for path, error in errors[:5]:
            click.echo(f"   - {path.name}: {error}")
        if len(errors) > 5:
            click.echo(f"   ... and {len(errors) - 5} more errors")
    
    if not dry_run and captions_generated > 0:
        click.echo(f"\nDONE! Captions saved to: {output_path}")
    elif dry_run:
        click.echo("\nDry run complete. No files were modified.")


if __name__ == "__main__":
    main()

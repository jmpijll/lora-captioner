"""
Command-line interface for LoRA Captioner.
"""

import click
from pathlib import Path

from lora_captioner import __version__


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
):
    """
    Caption images for LoRA training datasets.
    
    Uses Florence-2-PromptGen to generate training-ready captions
    optimized for different LoRA types (character, style, concept).
    """
    click.echo(f"LoRA Captioner v{__version__}")
    click.echo(f"Input: {input_path}")
    click.echo(f"Dataset name: {dataset_name}")
    click.echo(f"LoRA type: {lora_type}")
    click.echo(f"Trigger word: {trigger_word or '(none)'}")
    click.echo(f"Device: {device}")
    
    if dry_run:
        click.echo("\n[DRY RUN MODE - No changes will be made]")
    
    # TODO: Implement captioning pipeline
    click.echo("\n⚠️  Captioning pipeline not yet implemented.")
    click.echo("See PLAN.md for development roadmap.")


if __name__ == "__main__":
    main()

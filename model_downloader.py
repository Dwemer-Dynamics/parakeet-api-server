"""
Model downloader for Parakeet STT models
"""
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm
import config


class DownloadProgressBar:
    """Progress bar for downloads"""
    def __init__(self, desc="Downloading"):
        self.pbar = None
        self.desc = desc

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(
                total=total_size,
                unit='B',
                unit_scale=True,
                desc=self.desc
            )
        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size)
        else:
            self.pbar.close()


def download_file(url: str, destination: Path) -> None:
    """Download a file with progress bar"""
    print(f"Downloading: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        progress = DownloadProgressBar(desc=destination.name)
        urlretrieve(url, destination, reporthook=progress)
        print(f"✓ Downloaded: {destination.name}")
    except Exception as e:
        print(f"✗ Failed to download {destination.name}: {e}")
        if destination.exists():
            destination.unlink()
        raise


def download_onnx_models(precision: str) -> Path:
    """Download ONNX models as tar.bz2 archive and extract"""

    # Check if URL exists for this precision
    if precision not in config.ONNX_MODEL_URLS:
        raise ValueError(f"No {precision} ONNX model available")

    url = config.ONNX_MODEL_URLS[precision]
    model_dir_name = config.ONNX_MODEL_DIRS[precision]
    model_dir = config.MODELS_DIR / model_dir_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Determine required files based on precision
    if precision == 'int8':
        required_files = ['encoder.int8.onnx', 'decoder.int8.onnx', 'joiner.int8.onnx', 'tokens.txt']
    else:
        raise ValueError(f"Unsupported ONNX precision: {precision}")

    # Check if already downloaded
    all_exist = all((model_dir / f).exists() for f in required_files)

    if all_exist:
        print(f"\n✓ {precision.upper()} models already downloaded!")
        for f in required_files:
            print(f"  ✓ {f}")
        return model_dir

    print(f"\n{'='*60}")
    print(f"Downloading {config.MODEL_DESCRIPTION} ({precision.upper()})")
    print(f"{'='*60}\n")

    # Download tar.bz2 archive
    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "model.tar.bz2"

        print(f"Downloading {precision.upper()} model archive...")
        try:
            progress = DownloadProgressBar(desc=f"{precision.upper()} models")
            urlretrieve(url, archive_path, reporthook=progress)
        except Exception as e:
            print(f"\n✗ Failed to download {precision.upper()} models: {e}")
            sys.exit(1)

        # Extract archive
        print("\nExtracting model files...")
        try:
            with tarfile.open(archive_path, 'r:bz2') as tar:
                # Extract to temp dir first
                tar.extractall(temp_dir)

                # Find the extracted directory
                extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
                if not extracted_dirs:
                    raise Exception("No directory found in archive")

                extracted_dir = extracted_dirs[0]

                # Move files to model_dir
                for filename in required_files:
                    src = extracted_dir / filename
                    dst = model_dir / filename
                    if src.exists():
                        src.rename(dst)
                        print(f"  ✓ Extracted: {filename}")
                    else:
                        print(f"  ✗ Missing: {filename}")

        except Exception as e:
            print(f"\n✗ Failed to extract models: {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"✓ {precision.upper()} models downloaded successfully!")
    print(f"{'='*60}\n")

    return model_dir


def download_models(precision: str) -> Path:
    """
    Download Parakeet models

    Args:
        precision: Precision level ('fp32', 'int8')

    Returns:
        Path to model directory (only for ONNX models)
    """
    # Validate precision
    if precision not in config.AVAILABLE_PRECISION:
        raise ValueError(
            f"Invalid precision: {precision}. "
            f"Available: {', '.join(config.AVAILABLE_PRECISION)}"
        )

    # Only INT8 ONNX model needs to be downloaded
    # FP32 uses NeMo and is auto-downloaded from HuggingFace
    if precision == 'int8':
        return download_onnx_models(precision)
    else:
        # FP32 uses NeMo, no pre-download needed
        print(f"{precision.upper()} precision uses NeMo backend - model will be auto-downloaded from HuggingFace on first use")
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Parakeet STT models")
    parser.add_argument(
        "--precision",
        "-p",
        choices=config.AVAILABLE_PRECISION,
        default=config.DEFAULT_PRECISION,
        help=f"Precision level (default: {config.DEFAULT_PRECISION})"
    )

    args = parser.parse_args()

    # Show model info
    print(f"\nModel: {config.MODEL_DESCRIPTION}")
    print(f"Size: {config.MODEL_SIZE}")
    print(f"Languages: {config.MODEL_LANGUAGES}")
    print(f"Available precision: {', '.join(config.AVAILABLE_PRECISION)}")
    print()

    download_models(args.precision)

from pathlib import Path
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def is_blank_or_corrupt(image_path: Path) -> bool:
    try:
        with Image.open(image_path) as img:
            img.verify()
        with Image.open(image_path) as img:
            if img.getbbox() is None:
                return True
    except (OSError, UnidentifiedImageError):
        return True
    return False


def collect_input_images(input_path: Path):
    if input_path.is_dir():
        return sorted([p for p in input_path.iterdir() if is_image_file(p)])
    if input_path.is_file() and is_image_file(input_path):
        return [input_path]
    return []


def preprocess_images(input_path: Path, output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    images = collect_input_images(input_path)

    if not images:
        raise FileNotFoundError(f"No supported images found in {input_path}")

    kept = 0
    removed = 0
    for image_path in images:
        if is_blank_or_corrupt(image_path):
            print(f"Removed blank/corrupt image: {image_path.name}")
            removed += 1
            continue

        with Image.open(image_path) as img:
            img.convert("RGB").save(output_path / image_path.name)
        kept += 1

    print(f"Preprocessing complete: kept={kept}, removed={removed}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess images: remove blank or corrupt images.")
    parser.add_argument("--input", required=True, help="Input image file or folder")
    parser.add_argument("--output", required=True, help="Output folder for clean images")
    args = parser.parse_args()

    preprocess_images(Path(args.input), Path(args.output))

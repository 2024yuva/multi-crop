from pathlib import Path
import random
import shutil


SOURCE_ROOT = Path("data")
DEST_ROOT = Path("data_split")
SPLITS = ("train", "val", "test")
RATIOS = (0.7, 0.15, 0.15)
SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def iter_images(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.iterdir() if path.is_file() and path.suffix in IMAGE_EXTENSIONS
    )


def ensure_split_dirs(crop_name: str, class_names: list[str]) -> None:
    for split in SPLITS:
        for class_name in class_names:
            (DEST_ROOT / crop_name / split / class_name).mkdir(parents=True, exist_ok=True)


def split_class_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = files[:]
    random.shuffle(shuffled)

    train_end = int(len(shuffled) * RATIOS[0])
    val_end = train_end + int(len(shuffled) * RATIOS[1])

    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def main() -> None:
    random.seed(SEED)

    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(f"Source folder not found: {SOURCE_ROOT}")

    for crop_dir in sorted(path for path in SOURCE_ROOT.iterdir() if path.is_dir()):
        class_dirs = sorted(path for path in crop_dir.iterdir() if path.is_dir())
        class_names = [path.name for path in class_dirs]
        ensure_split_dirs(crop_dir.name, class_names)

        for class_dir in class_dirs:
            train_files, val_files, test_files = split_class_files(iter_images(class_dir))
            split_map = {
                "train": train_files,
                "val": val_files,
                "test": test_files,
            }

            for split_name, files in split_map.items():
                destination = DEST_ROOT / crop_dir.name / split_name / class_dir.name
                for file_path in files:
                    shutil.copy2(file_path, destination / file_path.name)

        print(f"Split complete for {crop_dir.name}")


if __name__ == "__main__":
    main()

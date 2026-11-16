"""Count and decode images without changing the dataset."""
from pathlib import Path
import hashlib
from PIL import Image

ROOT = Path(__file__).resolve().parent / "data" / "dataset-resized"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")


def main():
    total = 0
    failures = []
    seen = {}
    duplicates = []
    print("SmartSort AI - dataset check\n")
    for category in (*CLASSES, "trash"):
        folder = ROOT / category
        if not folder.is_dir():
            raise SystemExit(f"Missing category folder: {folder}")
        files = sorted(p for p in folder.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            failures.append(f"{category}: no images found")
        print(f"{category:12} {len(files):4} images"
              + (" (excluded from our model)" if category == "trash" else ""))
        if category in CLASSES:
            total += len(files)
        for path in files:
            try:
                with Image.open(path) as img:
                    rgb = img.convert("RGB")
                    rgb.load()
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                if category in CLASSES:
                    if digest in seen:
                        duplicates.append((str(seen[digest].relative_to(ROOT)),
                                           str(path.relative_to(ROOT))))
                    else:
                        seen[digest] = path
            except (OSError, ValueError) as exc:
                failures.append(f"{path.relative_to(ROOT)}: {exc}")
    print(f"\nFive-category total: {total}")
    print(f"Unreadable images / empty categories: {len(failures)}")
    print(f"Exact pixel duplicates in five categories: {len(duplicates)}")
    for item in failures:
        print("CHECK:", item)
    for first, second in duplicates:
        print(f"DUPLICATE: {first} = {second}")
    print("\nFiles were only checked; nothing was moved or deleted.")
    print("Next: review similar images and prepare train/validation/test splits.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

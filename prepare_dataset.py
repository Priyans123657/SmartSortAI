"""Create repeatable split lists; never move or delete source photographs.

Paths in the CSV files are relative to the SmartSortAI project folder.
Exact conflicting labels are excluded. Similar-image groups stay in one split.
The similarity screen is a heuristic, not a guarantee of no related images.
"""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
RAW = BASE / "data" / "dataset-resized"
OUT = BASE / "data" / "splits"
CLASSES = ("cardboard", "glass", "metal", "paper", "plastic")
RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
SEED = 42


def difference_hash(image):
    small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(small.get_flattened_data())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def write_csv(name, rows, fields):
    path = OUT / name
    temp = path.with_suffix(".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def main():
    print("Reading images. Please wait...", flush=True)
    buckets = defaultdict(list)
    for category in CLASSES:
        folder = RAW / category
        paths = sorted(p for p in folder.glob("*")
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not paths:
            raise SystemExit(f"Missing or empty category: {folder}")
        for path in paths:
            # Invalid images stop preparation rather than being silently skipped.
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                dhash = difference_hash(rgb)
            buckets[sha].append({"path": path.relative_to(BASE).as_posix(),
                                 "label": category, "sha256_pixels": sha,
                                 "dhash": dhash})

    kept, excluded = [], []
    for bucket in buckets.values():
        if len({row["label"] for row in bucket}) > 1:
            excluded.extend(dict(row, reason="identical_image_conflicting_labels")
                            for row in bucket)
        else:
            kept.append(bucket[0])
            excluded.extend(dict(row, reason="identical_image_same_label")
                            for row in bucket[1:])
    kept.sort(key=lambda row: row["path"])

    # Group likely near-duplicates across ALL categories before splitting.
    # Low 64-bit dHash distance can also group unrelated images; grouping only
    # restricts their split membership and never changes their labels.
    parent = list(range(len(kept)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    similar_pairs = []
    print("Screening similar images and making splits...", flush=True)
    for i, left in enumerate(kept):
        for j in range(i):
            distance = (left["dhash"] ^ kept[j]["dhash"]).bit_count()
            if distance <= 4:
                parent[root(i)] = root(j)
                similar_pairs.append({"first": kept[j]["path"],
                                      "second": left["path"], "distance": distance})
    groups = defaultdict(list)
    for i, row in enumerate(kept):
        groups[root(i)].append(row)
    group_list = list(groups.values())
    random.Random(SEED).shuffle(group_list)
    group_list.sort(key=len, reverse=True)
    totals = Counter(row["label"] for row in kept)
    counts = {split: Counter() for split in RATIOS}
    splits = {split: [] for split in RATIOS}
    for number, group in enumerate(group_list):
        sizes = Counter(row["label"] for row in group)

        def cost(split):
            # Change in squared class-count error; targets approximate 70/15/15.
            return sum(((counts[split][label] + size - RATIOS[split] * totals[label]) ** 2
                        - (counts[split][label] - RATIOS[split] * totals[label]) ** 2)
                       / totals[label] for label, size in sizes.items())

        chosen = min(RATIOS, key=cost)
        for row in group:
            row["group"] = f"g{number:05d}"
        splits[chosen].extend(group)
        counts[chosen].update(sizes)

    # Required integrity checks before writing any split files.
    assert sum(map(len, splits.values())) == len(kept)
    for field in ("path", "sha256_pixels", "group"):
        sets = {s: {row[field] for row in rows} for s, rows in splits.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            assert sets[a].isdisjoint(sets[b]), f"Leakage detected: {field}"
    for split in RATIOS:
        assert all(counts[split][label] > 0 for label in CLASSES), "Empty class in split"
    assert len(kept) + len(excluded) == sum(map(len, buckets.values()))

    OUT.mkdir(parents=True, exist_ok=True)
    fields = ["path", "label", "sha256_pixels", "group"]
    for split, rows in splits.items():
        write_csv(f"{split}.csv", sorted(rows, key=lambda row: row["path"]), fields)
    write_csv("excluded.csv", excluded, ["path", "label", "reason", "sha256_pixels"])
    write_csv("similar_pairs.csv", similar_pairs, ["first", "second", "distance"])
    summary = {"seed": SEED, "target_ratios": RATIOS,
               "classes": CLASSES, "input_count": len(kept) + len(excluded),
               "kept": len(kept), "excluded": len(excluded),
               "similarity_method": "64-bit dHash distance <= 4; transitive groups",
               "similar_pairs": len(similar_pairs),
               "counts": {s: dict(counts[s]) for s in RATIOS},
               "limitation": "Heuristic grouping cannot guarantee all same-object images are found."
               " Manual review and independent real-world testing remain necessary."}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nExcluded images: {len(excluded)} (original files preserved)")
    print(f"Usable images: {len(kept)}")
    for split in RATIOS:
        print(f"{split:12}: {len(splits[split])}")
    print(f"Similar-image pairs grouped: {len(similar_pairs)}")
    print("Split integrity checks: PASS")
    print("Saved lists to data/splits/. Photos have not been copied or deleted.")


if __name__ == "__main__":
    main()

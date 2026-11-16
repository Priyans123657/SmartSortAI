"""Fixed synthetic stress tests of the baseline, without changing source images."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from train import BASE, CLASSES, WasteDataset
from evaluate import classification_metrics, write_csv

# Fixed before viewing stress-test results; do not tune these using test scores.
CONDITIONS = [("clean", "clean", 0),
              ("blur_mild", "blur", 1), ("blur_medium", "blur", 2), ("blur_strong", "blur", 4),
              ("dark_mild", "dark", 0.7), ("dark_medium", "dark", 0.4), ("dark_strong", "dark", 0.2),
              ("occlusion_mild", "occlusion", 0.1),
              ("occlusion_medium", "occlusion", 0.2),
              ("occlusion_strong", "occlusion", 0.3)]


def corrupt(image, kind, value):
    if kind == "clean":
        return image.copy()
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=value))
    if kind == "dark":
        return ImageEnhance.Brightness(image).enhance(value)
    if kind == "occlusion":
        result = image.copy()
        side = round(math.sqrt(value * image.width * image.height))
        left, top = (image.width - side) // 2, (image.height - side) // 2
        result.paste((127, 127, 127), (left, top, left + side, top + side))
        return result
    raise ValueError(kind)


class StressDataset(Dataset):
    def __init__(self, rows, images, kind, value):
        self.rows, self.images, self.kind, self.value = rows, images, kind, value
        weights_transform = ResNet18_Weights.DEFAULT.transforms()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(weights_transform.mean, weights_transform.std)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        modified = corrupt(self.images[index], self.kind, self.value)
        return self.to_tensor(modified), CLASSES.index(self.rows[index]["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = sorted((BASE / "models").glob("baseline_*/best.pt"))
        if len(candidates) != 1:
            raise SystemExit("Supply --checkpoint when there is not exactly one baseline model.")
        checkpoint_path = candidates[0]
    checkpoint_path = checkpoint_path.resolve()
    result_root = BASE / "results" / checkpoint_path.parent.name
    clean_path = result_root / "clean_test" / "metrics.json"
    if not clean_path.exists():
        raise SystemExit("Run evaluate.py for this checkpoint first.")
    clean = json.loads(clean_path.read_text(encoding="utf-8"))
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != clean["checkpoint_sha256"]:
        raise SystemExit("Checkpoint changed since clean evaluation.")
    torch.set_num_threads(4)
    test = WasteDataset("test")
    if hashlib.sha256(test.manifest.read_bytes()).hexdigest() != clean["test_manifest_sha256"]:
        raise SystemExit("Test manifest changed since clean evaluation.")
    weights_transform = ResNet18_Weights.DEFAULT.transforms()
    geometry = transforms.Compose([
        transforms.Resize(weights_transform.resize_size, interpolation=weights_transform.interpolation,
                          antialias=weights_transform.antialias),
        transforms.CenterCrop(weights_transform.crop_size)])
    images = []
    for row in test.rows:
        with Image.open(BASE / row["path"]) as source:
            rgb = source.convert("RGB")
            sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if sha != row["sha256_pixels"]:
                raise SystemExit(f"Test photo changed: {row['path']}")
            images.append(geometry(rgb))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise SystemExit("Class order mismatch.")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    out = result_root / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    results, summary, all_predictions = [], [], []
    print(f"Testing {len(test)} images in each of {len(CONDITIONS)} conditions on {device}.", flush=True)
    with torch.inference_mode():
        for name, kind, value in CONDITIONS:
            dataset = StressDataset(test.rows, images, kind, value)
            loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)
            matrix = torch.zeros((5, 5), dtype=torch.int64)
            offset = 0
            for batch, labels in loader:
                logits = model(batch.to(device))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite predictions.")
                confidence, predicted = logits.softmax(1).max(1)
                predicted, confidence = predicted.cpu(), confidence.cpu()
                matrix += torch.bincount(labels * 5 + predicted, minlength=25).reshape(5, 5)
                for j in range(len(labels)):
                    all_predictions.append(dict(condition=name, path=test.rows[offset + j]["path"],
                                                actual=CLASSES[int(labels[j])], predicted=CLASSES[int(predicted[j])],
                                                uncalibrated_confidence=float(confidence[j])))
                offset += len(labels)
            assert int(matrix.sum()) == len(test)
            if kind == "clean" and matrix.tolist() != clean["confusion_matrix"]:
                raise RuntimeError("Clean control differs from previous evaluation. Diagnose preprocessing first.")
            metric = classification_metrics(matrix)
            metric.update(condition=name, kind=kind, parameter=value, confusion_matrix=matrix.tolist())
            results.append(metric)
            summary.append(dict(condition=name, correct=metric["correct"], total=metric["total"],
                                accuracy=metric["accuracy"], macro_f1=metric["macro_f1"],
                                accuracy_drop_pp=100 * (clean["accuracy"] - metric["accuracy"])))
            print(f"{name:20} accuracy={metric['accuracy']:.2%}  macro F1={metric['macro_f1']:.4f}", flush=True)
    protocol = dict(checkpoint_sha256=clean["checkpoint_sha256"],
                    test_manifest_sha256=clean["test_manifest_sha256"],
                    image_space="Resize short side to 256, center crop 224, corrupt, tensor/normalize",
                    blur="PIL Gaussian blur radius 1/2/4 pixels on 224x224 crop",
                    darkness="Brightness multipliers 0.7/0.4/0.2; no noise simulation",
                    occlusion="Centered gray square covering approximately 10/20/30 percent of crop area",
                    limitations="Synthetic central masking is not real multi-object overlap. Conditions are separate,"
                    " not combined. Same 357 images recur; 3570 predictions are not 3570 independent images.",
                    confidence="uncalibrated; no review threshold selected on test data")
    (out / "metrics.json").write_text(json.dumps(dict(protocol=protocol, conditions=results), indent=2), encoding="utf-8")
    write_csv(out / "summary.csv", summary, list(summary[0]))
    write_csv(out / "predictions.csv", all_predictions, list(all_predictions[0]))
    rows = "".join(f"<tr><th>{r['condition']}</th><td>{r['correct']}/{r['total']}</td>"
                   f"<td>{r['accuracy']:.2%}</td><td>{r['macro_f1']:.4f}</td>"
                   f"<td>{r['accuracy_drop_pp']:.2f}</td></tr>" for r in summary)
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>SmartSort robustness</title>
<style>body{{font:17px system-ui;max-width:1050px;margin:40px auto;padding:20px;color:#183b32;background:#f6faf8}}
table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border:1px solid #cadbd2;text-align:left}}
thead{{background:#deeee5}}p{{line-height:1.6}}</style><h1>SmartSort AI: challenging conditions</h1>
<p>Same saved baseline, same {len(test)} held-out images in every condition. No retraining.</p>
<table><thead><tr><th>Condition</th><th>Correct</th><th>Accuracy</th><th>Macro F1</th>
<th>Accuracy drop (percentage points)</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Experiment settings</h2><p>{html.escape(protocol['image_space'])}.</p>
<p>Blur: {html.escape(protocol['blur'])}.<br>Darkness: {html.escape(protocol['darkness'])}.<br>
Occlusion: {html.escape(protocol['occlusion'])}.</p>
<p>{html.escape(protocol['limitations'])}</p><p>Confidence calibration and review flags are pending.
Use validation data for future model choices and thresholds; do not tune on these test scores.</p></html>"""
    (out / "report.html").write_text(document, encoding="utf-8")
    print("\nRobustness evaluation complete. Original photos and model unchanged.")
    print("Report:", out / "report.html")


if __name__ == "__main__":
    main()

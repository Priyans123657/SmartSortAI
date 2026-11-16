"""Evaluate a saved baseline on the held-out clean test split; no training."""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from train import BASE, CLASSES, WasteDataset


def classification_metrics(matrix):
    rows = []
    for i, label in enumerate(CLASSES):
        tp = int(matrix[i, i])
        support = int(matrix[i].sum())
        predicted = int(matrix[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        rows.append(dict(label=label, precision=precision, recall=recall,
                         f1=f1, support=support))
    total = int(matrix.sum())
    correct = int(matrix.diag().sum())
    return dict(total=total, correct=correct, accuracy=correct / total,
                macro_f1=sum(r["f1"] for r in rows) / len(CLASSES), per_class=rows)


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_confusion_table(path, matrix, result):
    headers = "".join(f"<th scope='col'>{name}</th>" for name in CLASSES)
    table_rows = []
    for i, label in enumerate(CLASSES):
        cells = "".join(f"<td class='{'correct' if i == j else 'error'}'>{int(matrix[i, j])}</td>"
                        for j in range(len(CLASSES)))
        table_rows.append(f"<tr><th scope='row'>{label}</th>{cells}</tr>")
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>SmartSort AI - Clean test confusion matrix</title>
<style>body{{font:17px system-ui,sans-serif;max-width:1000px;margin:48px auto;padding:20px;color:#173128;background:#f6f9f7}}
table{{border-collapse:collapse;background:white;width:100%;margin:24px 0}}
th,td{{padding:16px;border:1px solid #c7d8cf;text-align:center}}
th{{background:#e4eee8}}.correct{{background:#d1efda;font-weight:700}}.error{{background:#fff8ed}}
p{{line-height:1.6}}</style><h1>SmartSort AI: clean test results</h1>
<p>Accuracy: <b>{result['accuracy']:.2%}</b> &nbsp; Macro F1: <b>{result['macro_f1']:.4f}</b><br>
Correct: {result['correct']} / {result['total']} images. Saved epoch: {result['epoch']}.</p>
<h2>Confusion matrix</h2><p>Rows = actual category. Columns = predicted category.
Green diagonal cells are correct predictions; other cells are mistakes.</p>
<table><thead><tr><th>Actual / Predicted</th>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody></table>
<p>Model: {html.escape(result['checkpoint'])}</p>
<p>This evaluates clean TrashNet images only. Blur, low light, occlusion and independent
real-world testing remain separate experiments. Confidence is uncalibrated.
Do not use these test results to tune model settings or confidence thresholds.</p></html>"""
    path.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Explicit best.pt path if multiple runs exist")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = sorted((BASE / "models").glob("baseline_*/best.pt"))
        if len(candidates) != 1:
            raise SystemExit("Expected one baseline checkpoint. Supply --checkpoint with its best.pt path.")
        checkpoint_path = candidates[0]
    checkpoint_path = checkpoint_path.resolve()
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES or checkpoint["config"]["architecture"] != "resnet18":
        raise SystemExit("Checkpoint architecture or label order does not match.")
    test = WasteDataset("test")
    for split in ("train", "validation"):
        manifest = BASE / "data" / "splits" / f"{split}.csv"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != checkpoint["config"]["split_hashes"][split]:
            raise SystemExit(f"{split} manifest changed since training. Stop and diagnose.")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for field in ("path", "sha256_pixels", "group"):
            if not {r[field] for r in rows}.isdisjoint({r[field] for r in test.rows}):
                raise SystemExit(f"Test overlaps {split}: {field}")
    # Confirm test photos still match the audited manifest before scoring.
    from PIL import Image
    for row in test.rows:
        with Image.open(BASE / row["path"]) as image:
            rgb = image.convert("RGB")
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        if digest != row["sha256_pixels"]:
            raise SystemExit(f"Test image changed: {row['path']}")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    print(f"Evaluating saved epoch {checkpoint['epoch']} on {len(test)} test images ({device}).", flush=True)
    loader = DataLoader(test, batch_size=32, shuffle=False, num_workers=0)
    matrix = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    predictions = []
    offset = 0
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            if not torch.isfinite(logits).all():
                raise RuntimeError("Non-finite predictions.")
            confidence, predicted = logits.softmax(1).max(1)
            predicted = predicted.cpu()
            indices = labels * len(CLASSES) + predicted
            matrix += torch.bincount(indices, minlength=len(CLASSES) ** 2).reshape(matrix.shape)
            for j in range(len(labels)):
                predictions.append(dict(path=test.rows[offset + j]["path"], actual=CLASSES[int(labels[j])],
                                        predicted=CLASSES[int(predicted[j])],
                                        uncalibrated_confidence=float(confidence[j]),
                                        correct=bool(labels[j] == predicted[j])))
            offset += len(labels)
    assert int(matrix.sum()) == len(test) == len(predictions)
    result = classification_metrics(matrix)
    result.update(checkpoint=str(checkpoint_path), epoch=checkpoint["epoch"],
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  test_manifest_sha256=hashlib.sha256(test.manifest.read_bytes()).hexdigest(),
                  classes=CLASSES, confusion_matrix=matrix.tolist(),
                  condition="clean", confidence="uncalibrated")
    out = BASE / "results" / checkpoint_path.parent.name / "clean_test"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_csv(out / "per_class.csv", result["per_class"], ["label", "precision", "recall", "f1", "support"])
    write_csv(out / "predictions.csv", predictions, list(predictions[0]))
    write_csv(out / "confusion_matrix.csv",
              [dict(actual=label, **{name: int(matrix[i, j]) for j, name in enumerate(CLASSES)})
               for i, label in enumerate(CLASSES)], ["actual", *CLASSES])
    save_confusion_table(out / "confusion_matrix.html", matrix, result)
    print(f"Test accuracy: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")
    print(f"Test macro F1: {result['macro_f1']:.4f}")
    print("\nCategory       Precision  Recall     F1")
    for row in result["per_class"]:
        print(f"{row['label']:14} {row['precision']:.3f}      {row['recall']:.3f}      {row['f1']:.3f}")
    print("\nEvaluation complete. No model weights changed.")
    print("Results:", out)
    print("Open confusion_matrix.html in a browser to view the table.")
    print("These are clean-dataset results; challenging-condition evaluation is still pending.")


if __name__ == "__main__":
    main()

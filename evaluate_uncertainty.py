"""Test the fixed validation-derived review rule. Never fit a threshold here."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

from train import BASE, CLASSES, WasteDataset
from evaluate import write_csv
from evaluate_robustness import CONDITIONS, StressDataset
from preprocessing import geometry_for


def review_metrics(correct, accepted):
    total = len(correct)
    count = int(accepted.sum())
    errors = int((~correct).sum())
    accepted_errors = int((accepted & ~correct).sum())
    flagged_errors = errors - accepted_errors
    return dict(total=total, overall_accuracy=float(correct.double().mean()),
                accepted=count, review=total-count, coverage=count / total,
                review_rate=1-count / total,
                accepted_accuracy=(count-accepted_errors) / count if count else None,
                total_errors=errors, accepted_errors=accepted_errors,
                flagged_errors=flagged_errors,
                error_capture_rate=flagged_errors / errors if errors else None,
                correct_sent_for_review=int((~accepted & correct).sum()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-frame", action="store_true")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = sorted((BASE / "models").glob("baseline_*/best.pt"))
        if len(candidates) != 1:
            raise SystemExit("Supply --checkpoint when there is not exactly one baseline.")
        checkpoint_path = candidates[0]
    checkpoint_path = checkpoint_path.resolve()
    policy_path = checkpoint_path.parent / ("uncertainty_full_frame.json" if args.full_frame else "uncertainty.json")
    if not policy_path.exists():
        raise SystemExit("Run calibrate.py first.")
    policy_bytes = policy_path.read_bytes()
    policy = json.loads(policy_bytes)
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    if checkpoint_hash != policy["checkpoint_sha256"] or policy["classes"] != CLASSES:
        raise SystemExit("Policy does not belong to this model/class order.")
    temperature, threshold = policy["temperature"], policy["threshold"]
    if not (math.isfinite(temperature) and temperature > 0 and math.isfinite(threshold) and 0 <= threshold <= 1.01):
        raise SystemExit("Invalid policy parameters.")
    if policy["conditions"] != [list(c) for c in CONDITIONS]:
        raise SystemExit("Condition definitions changed since calibration.")
    torch.set_num_threads(4)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    test = WasteDataset("test")
    for split in ("train", "validation"):
        manifest = BASE / "data" / "splits" / f"{split}.csv"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != checkpoint["config"]["split_hashes"][split]:
            raise SystemExit(f"{split} manifest changed since training.")
        if split == "validation" and digest != policy["validation_manifest_sha256"]:
            raise SystemExit("Validation changed since calibration.")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for field in ("path", "sha256_pixels", "group"):
            if not {r[field] for r in test.rows}.isdisjoint({r[field] for r in rows}):
                raise SystemExit(f"Overlap between test and {split}: {field}")
    wt = ResNet18_Weights.DEFAULT.transforms()
    view = "full_frame" if args.full_frame else "center_crop"
    if policy.get("preprocessing", "center_crop") != view:
        raise SystemExit("Preprocessing and calibration policy mismatch.")
    geometry = geometry_for(view)
    images = []
    for row in test.rows:
        with Image.open(BASE / row["path"]) as source:
            rgb = source.convert("RGB")
            sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if sha != row["sha256_pixels"]:
                raise SystemExit(f"Test image changed: {row['path']}")
            images.append(geometry(rgb))
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    results, predictions, all_correct, all_accepted = [], [], [], []
    print(f"Fixed temperature: {temperature:.4f}; fixed threshold: {threshold:.4f}", flush=True)
    with torch.inference_mode():
        for name, kind, value in CONDITIONS:
            loader = DataLoader(StressDataset(test.rows, images, kind, value),
                                batch_size=32, shuffle=False, num_workers=0)
            correct_list, accepted_list = [], []
            offset = 0
            for batch, labels in loader:
                logits = model(batch.to(device))
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite predictions.")
                confidence, predicted = (logits / temperature).softmax(1).max(1)
                predicted, confidence = predicted.cpu(), confidence.cpu()
                correct, accepted = predicted == labels, confidence >= threshold
                correct_list.append(correct)
                accepted_list.append(accepted)
                for j in range(len(labels)):
                    predictions.append(dict(condition=name, path=test.rows[offset+j]["path"],
                                            actual=CLASSES[int(labels[j])], predicted=CLASSES[int(predicted[j])],
                                            calibrated_confidence=float(confidence[j]),
                                            decision="Accepted" if accepted[j] else "Manual review needed",
                                            correct=bool(correct[j])))
                offset += len(labels)
            correct, accepted = torch.cat(correct_list), torch.cat(accepted_list)
            assert len(correct) == len(test)
            stats = dict(condition=name, **review_metrics(correct, accepted))
            results.append(stats)
            all_correct.append(correct)
            all_accepted.append(accepted)
            accuracy_text = f"{stats['accepted_accuracy']:.1%}" if stats['accepted_accuracy'] is not None else "N/A"
            print(f"{name:20} accepted={stats['accepted']:3} review={stats['review']:3} "
                  f"accepted accuracy={accuracy_text} accepted errors={stats['accepted_errors']}", flush=True)
    pooled = dict(condition="pooled_equal_conditions", **review_metrics(torch.cat(all_correct), torch.cat(all_accepted)))
    assert policy_path.read_bytes() == policy_bytes, "Policy changed during evaluation"
    out = BASE / "results" / checkpoint_path.parent.name / ("uncertainty_test_full_frame" if args.full_frame else "uncertainty_test")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary.csv", [*results, pooled], list(pooled))
    write_csv(out / "predictions.csv", predictions, list(predictions[0]))
    report = dict(temperature=temperature, threshold=threshold, preprocessing=view, conditions=results, pooled=pooled,
                  checkpoint_sha256=checkpoint_hash, policy_sha256=hashlib.sha256(policy_bytes).hexdigest(),
                  test_manifest_sha256=hashlib.sha256(test.manifest.read_bytes()).hexdigest(),
                  limitations="Pooled statistics weight each synthetic condition equally, not expected real-world prevalence."
                  " 3570 predictions reuse 357 images. Accepted accuracy excludes review cases and is not overall accuracy."
                  " No human review outcomes were simulated. The validation 95% target is not guaranteed on test data.")
    (out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    def percent(value):
        return "N/A" if value is None else f"{value:.2%}"
    rows_html = "".join(f"<tr><th>{r['condition']}</th><td>{r['accepted']}</td><td>{r['review']}</td>"
                        f"<td>{percent(r['coverage'])}</td><td>{percent(r['accepted_accuracy'])}</td>"
                        f"<td>{r['accepted_errors']}</td><td>{r['flagged_errors']}/{r['total_errors']}</td></tr>"
                        for r in [*results, pooled])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><title>SmartSort review policy</title>
<style>body{{font:17px system-ui;max-width:1200px;margin:40px auto;padding:20px;background:#f6faf8;color:#183b32}}
table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:12px;border:1px solid #cadbd2;text-align:left}}
thead{{background:#deeee5}}p{{line-height:1.6}}</style><h1>SmartSort AI: review rule on test images</h1>
<p>Temperature {temperature:.4f}; confidence threshold {threshold:.4f}. Both fixed using validation data.</p>
<table><thead><tr><th>Condition</th><th>Accepted</th><th>Review</th><th>Coverage</th>
<th>Accepted accuracy</th><th>Wrong but accepted</th><th>Errors flagged / all errors</th></tr></thead>
<tbody>{rows_html}</tbody></table><p>Coverage = accepted predictions / all predictions.
Accepted accuracy = correct accepted predictions / accepted predictions.
Review includes both wrong and correct predictions below the confidence threshold.</p>
<p>{report['limitations']}</p><p>Do not adjust the threshold using this report.</p></html>"""
    (out / "report.html").write_text(document, encoding="utf-8")
    print("\nUncertainty evaluation complete. Threshold and model unchanged.")
    print("Report:", out / "report.html")


if __name__ == "__main__":
    main()

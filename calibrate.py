"""Fit temperature and a review policy using validation data only.

The validation set was previously used for epoch selection, so these are tuning
statistics, not independent performance estimates. Test evaluation comes later.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random

from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from train import BASE, CLASSES, WasteDataset
from evaluate_robustness import CONDITIONS, StressDataset
from evaluate import write_csv
from preprocessing import geometry_for


def select_policy(confidence, correct, target=0.95, min_coverage=0.10):
    """Maximize coverage subject to an empirical pooled accuracy target.

    Threshold 1.01 is an explicit review-all fallback when the target is infeasible.
    This is not a statistical guarantee of accuracy on future examples.
    """
    minimum = max(20, math.ceil(len(correct) * min_coverage))
    chosen = {"threshold": 1.01, "accepted": 0, "coverage": 0.0,
              "accepted_accuracy": None, "target_met": False}
    for threshold in sorted(set([0.0] + confidence.tolist())):
        mask = confidence >= threshold
        count = int(mask.sum())
        if count < minimum:
            continue
        accuracy = float(correct[mask].float().mean())
        if accuracy >= target and count > chosen["accepted"]:
            chosen = dict(threshold=threshold, accepted=count,
                          coverage=count / len(correct), accepted_accuracy=accuracy, target_met=True)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--full-frame", action="store_true", help="Calibrate aspect-preserving whole-image input separately")
    args = parser.parse_args()
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        candidates = sorted((BASE / "models").glob("baseline_*/best.pt"))
        if len(candidates) != 1:
            raise SystemExit("Specify --checkpoint when there is not exactly one baseline.")
        checkpoint_path = candidates[0]
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise SystemExit("Class order mismatch.")
    torch.set_num_threads(4)
    validation = WasteDataset("validation")
    validation_hash = hashlib.sha256(validation.manifest.read_bytes()).hexdigest()
    if validation_hash != checkpoint["config"]["split_hashes"]["validation"]:
        raise SystemExit("Validation split changed since training.")

    # Hold groups and every degraded version of a source together.
    groups = sorted({row["group"] for row in validation.rows})
    random.Random(2026).shuffle(groups)
    fit_groups = set(groups[:len(groups) // 2])
    fit_source = torch.tensor([row["group"] in fit_groups for row in validation.rows])
    assert fit_source.any() and (~fit_source).any()
    wt = ResNet18_Weights.DEFAULT.transforms()
    view = "full_frame" if args.full_frame else "center_crop"
    geometry = geometry_for(view)
    images = []
    for row in validation.rows:
        with Image.open(BASE / row["path"]) as source:
            rgb = source.convert("RGB")
            sha = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if sha != row["sha256_pixels"]:
                raise SystemExit(f"Validation image changed: {row['path']}")
            images.append(geometry(rgb))
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    logits_list, labels_list = [], []
    print("Validation only: collecting clean and degraded predictions...", flush=True)
    with torch.inference_mode():
        for name, kind, value in CONDITIONS:
            loader = DataLoader(StressDataset(validation.rows, images, kind, value),
                                batch_size=32, shuffle=False, num_workers=0)
            for batch, labels in loader:
                logits_list.append(model(batch.to(device)).cpu())
                labels_list.append(labels)
            print("  Completed:", name, flush=True)
    logits, labels = torch.cat(logits_list), torch.cat(labels_list)
    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite predictions.")
    fit_mask = fit_source.repeat(len(CONDITIONS))
    # Grid includes identity (T=1), so fit NLL cannot be worse than unscaled NLL.
    candidates = sorted(set([1.0] + torch.logspace(-0.30103, 1, 161).tolist()))
    losses = [float(nn.functional.cross_entropy(logits[fit_mask] / t, labels[fit_mask])) for t in candidates]
    temperature = candidates[min(range(len(losses)), key=losses.__getitem__)]
    confidence, predicted = (logits / temperature).softmax(1).max(1)
    assert torch.equal(predicted, logits.argmax(1)), "Temperature changed class predictions"
    correct = predicted == labels
    policy = select_policy(confidence[~fit_mask], correct[~fit_mask])
    selected = confidence >= policy["threshold"]
    diagnostics = []
    n = len(validation)
    for index, (name, _, _) in enumerate(CONDITIONS):
        condition_mask = torch.zeros(len(labels), dtype=torch.bool)
        condition_mask[index * n:(index + 1) * n] = ~fit_source
        accepted = selected & condition_mask
        total, count = int(condition_mask.sum()), int(accepted.sum())
        diagnostics.append(dict(condition=name, total=total, accepted=count, review=total-count,
                                coverage=count / total,
                                accepted_accuracy=float(correct[accepted].float().mean()) if count else None))
    nll = lambda mask, t: float(nn.functional.cross_entropy(logits[mask] / t, labels[mask]))
    result = dict(temperature=temperature, **policy, classes=CLASSES, preprocessing=view,
                  checkpoint=str(checkpoint_path),
                  checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
                  validation_manifest_sha256=validation_hash, target_accepted_accuracy=0.95,
                  minimum_coverage=0.10, minimum_accepted_predictions=20,
                  confidence_rule="softmax(logits / temperature).max; accept if >= threshold",
                  calibration_source_images=int(fit_source.sum()),
                  policy_source_images=int((~fit_source).sum()),
                  calibration_paths=[r["path"] for i, r in enumerate(validation.rows) if fit_source[i]],
                  policy_paths=[r["path"] for i, r in enumerate(validation.rows) if not fit_source[i]],
                  conditions=CONDITIONS,
                  fit_nll_before=nll(fit_mask, 1), fit_nll_after=nll(fit_mask, temperature),
                  policy_nll_before=nll(~fit_mask, 1), policy_nll_after=nll(~fit_mask, temperature),
                  per_condition_tuning_statistics=diagnostics,
                  limitations="Validation was also used for epoch selection. Calibration and policy halves"
                  " are group-disjoint, but reported policy statistics are used for threshold tuning."
                  " Equal-weight mixture of ten synthetic conditions; repeated source images are dependent."
                  " Pooled 95% target is empirical, not a guarantee, and need not hold for each condition."
                  " Softmax confidence cannot reliably reject every unknown material or wrong prediction.",
                  method_reference="https://proceedings.mlr.press/v70/guo17a.html")
    assert result["fit_nll_after"] <= result["fit_nll_before"] + 1e-6
    assert set(result["calibration_paths"]).isdisjoint(result["policy_paths"])
    policy_path = checkpoint_path.parent / ("uncertainty_full_frame.json" if args.full_frame else "uncertainty.json")
    policy_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    out = BASE / "results" / checkpoint_path.parent.name / ("calibration_full_frame" if args.full_frame else "calibration")
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "validation_policy.csv", diagnostics, list(diagnostics[0]))
    (out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nTemperature: {temperature:.4f}")
    if policy["target_met"]:
        print(f"Review threshold: {policy['threshold']:.4f}")
        print(f"Accepted coverage on policy-tuning data: {policy['coverage']:.1%}")
        print(f"Accepted accuracy on policy-tuning data: {policy['accepted_accuracy']:.1%}")
    else:
        print("Target not met with minimum coverage. Policy sends all predictions for review.")
    print("Calibration complete. Saved:", policy_path)
    print("Test images were not loaded. Independent policy evaluation is still pending.")


if __name__ == "__main__":
    main()

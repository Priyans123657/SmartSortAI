"""Train a clean baseline or an augmented model. Test images are never loaded here."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

BASE = Path(__file__).resolve().parent
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic"]


class WasteDataset(Dataset):
    def __init__(self, split, augmented=False):
        if augmented and split != "train":
            raise ValueError("Random augmentation is allowed only for training.")
        self.manifest = BASE / "data" / "splits" / f"{split}.csv"
        with self.manifest.open(encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Empty split: {split}")
        self.transform = ResNet18_Weights.DEFAULT.transforms()
        if augmented:
            weights = ResNet18_Weights.DEFAULT.transforms()
            self.transform = transforms.Compose([
                transforms.Resize(256, interpolation=weights.interpolation, antialias=True),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=(0.2, 1.2), contrast=(0.7, 1.3),
                                       saturation=(0.8, 1.2)),
                transforms.RandomApply([transforms.GaussianBlur(13, sigma=(0.1, 2.0))], p=0.35),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.25), ratio=(0.5, 2.0), value=0.5),
                transforms.Normalize(weights.mean, weights.std),
            ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(BASE / row["path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, CLASSES.index(row["label"])


def metrics(matrix):
    matrix = matrix.float()
    true_positive = matrix.diag()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(denominator > 0, 2 * true_positive / denominator, 0)
    return (true_positive.sum() / matrix.sum()).item(), f1.mean().item()


def run_epoch(model, loader, device, optimizer=None, smoke=False):
    training = optimizer is not None
    model.train(training)
    loss_sum, count = 0.0, 0
    matrix = torch.zeros((len(CLASSES), len(CLASSES)), dtype=torch.int64)
    for step, (images, labels) in enumerate(loader, start=1):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; stop and diagnose before continuing.")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        count += batch_size
        indices = (labels * len(CLASSES) + logits.argmax(1)).detach().cpu()
        matrix += torch.bincount(indices, minlength=len(CLASSES) ** 2).reshape(matrix.shape)
        if smoke:
            break
        if training and (step % 20 == 0 or step == len(loader)):
            print(f"  Training batch {step}/{len(loader)}", flush=True)
    accuracy, macro_f1 = metrics(matrix)
    return {"loss": loss_sum / count, "accuracy": accuracy, "macro_f1": macro_f1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--augmented", action="store_true", help="Train a separate model with image-quality augmentation")
    parser.add_argument("--smoke", action="store_true", help="Offline one-batch code check, not real training")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 2:
        parser.error("Use epochs >= 1 and batch-size >= 2.")
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise SystemExit("GPU unavailable. Run check_setup.py and diagnose before training.")
    device = torch.device("cuda")
    print("Device:", torch.cuda.get_device_name(0), flush=True)
    datasets = {name: WasteDataset(name, augmented=args.augmented and name == "train")
                for name in ("train", "validation")}
    print("Experiment:", "augmented" if args.augmented else "clean baseline", flush=True)
    for field in ("path", "sha256_pixels", "group"):
        assert {r[field] for r in datasets["train"].rows}.isdisjoint(
            {r[field] for r in datasets["validation"].rows}), f"Split overlap: {field}"
    loaders = {name: DataLoader(dataset, batch_size=args.batch_size,
                               shuffle=name == "train", num_workers=0)
               for name, dataset in datasets.items()}
    print("Training images:", len(datasets["train"]), "| Validation images:",
          len(datasets["validation"]), flush=True)
    torch.hub.set_dir(str(BASE / "models" / "pretrained"))
    print("Offline smoke check (random weights)." if args.smoke else
          "Loading pretrained ResNet18; first run may download about 45 MB.", flush=True)
    model = resnet18(weights=None if args.smoke else ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.to(device)
    # Fine-tune the pretrained backbone slowly; learn the new five-class head faster.
    optimizer = torch.optim.AdamW([
        {"params": [p for name, p in model.named_parameters() if not name.startswith("fc.")], "lr": 1e-4},
        {"params": model.fc.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4)
    if args.smoke:
        before = model.fc.weight.detach().clone()
        run_epoch(model, loaders["train"], device, optimizer, smoke=True)
        assert not torch.equal(before, model.fc.weight), "Optimizer did not update weights"
        run_epoch(model, loaders["validation"], device, smoke=True)
        assert metrics(torch.eye(5, dtype=torch.int64)) == (1.0, 1.0)
        assert metrics(torch.ones((5, 5), dtype=torch.int64))[1] > 0
        print("SMOKE CHECK PASS: GPU forward, backward, weight update and validation.")
        print("No trained model saved. This check does not measure classification quality.")
        return

    experiment = "augmented" if args.augmented else "baseline"
    run_dir = BASE / "models" / (experiment + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {"architecture": "resnet18", "classes": CLASSES, "seed": 42,
              "epochs": args.epochs, "batch_size": args.batch_size,
              "optimizer": "AdamW", "backbone_lr": 1e-4, "head_lr": 1e-3,
              "weight_decay": 1e-4, "experiment": experiment,
              "augmentation": str(datasets["train"].transform) if args.augmented else "none (clean baseline)",
              "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
              "selection_metric": "validation macro F1", "torch": str(torch.__version__),
              "torchvision": str(torchvision.__version__),
              "split_hashes": {s: hashlib.sha256(d.manifest.read_bytes()).hexdigest()
                               for s, d in datasets.items()}}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    best_f1 = -1.0
    history = []
    print("Saving this run to:", run_dir, flush=True)
    for epoch in range(1, args.epochs + 1):
        start = time.monotonic()
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        train = run_epoch(model, loaders["train"], device, optimizer)
        validation = run_epoch(model, loaders["validation"], device)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train.items()},
               **{f"validation_{k}": v for k, v in validation.items()},
               "seconds": time.monotonic() - start}
        history.append(row)
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"Train accuracy: {train['accuracy']:.1%} | Validation accuracy: "
              f"{validation['accuracy']:.1%} | Validation macro F1: {validation['macro_f1']:.4f}", flush=True)
        if validation["macro_f1"] > best_f1:
            best_f1 = validation["macro_f1"]
            checkpoint = {"model_state_dict": model.state_dict(), "classes": CLASSES,
                          "epoch": epoch, "validation_metrics": validation, "config": config}
            temporary = run_dir / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(run_dir / "best.pt")
            print("Best model saved.", flush=True)
    print(f"\nTraining complete. Best validation macro F1: {best_f1:.4f}")
    print("Model:", run_dir / "best.pt")
    print("Test evaluation and confidence calibration are still pending.")


if __name__ == "__main__":
    main()

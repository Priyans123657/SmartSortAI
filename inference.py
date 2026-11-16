"""Shared inference for the local app; uses the saved calibration policy."""
import hashlib
import json
import math
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from train import CLASSES
from evaluate_robustness import CONDITIONS, corrupt
from preprocessing import geometry_for


def load_predictor(checkpoint_path, policy_filename="uncertainty.json"):
    path = Path(checkpoint_path)
    policy = json.loads(path.with_name(policy_filename).read_text(encoding="utf-8"))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != policy["checkpoint_sha256"] or policy["classes"] != CLASSES:
        raise ValueError("Model and uncertainty policy do not match. Run calibration for this model.")
    if not (math.isfinite(policy["temperature"]) and policy["temperature"] > 0
            and math.isfinite(policy["threshold"]) and 0 <= policy["threshold"] <= 1.01):
        raise ValueError("Invalid calibration settings.")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["classes"] != CLASSES:
        raise ValueError("Model label order does not match.")
    torch.set_num_threads(4)
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    return model, device, policy


def prepare_image(image, condition="clean", view="center_crop"):
    wt = ResNet18_Weights.DEFAULT.transforms()
    # Handle phone rotation and transparent images before model preprocessing.
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    geometry = geometry_for(view)
    lookup = {name: (kind, value) for name, kind, value in CONDITIONS}
    if condition not in lookup:
        raise ValueError("Unknown image condition.")
    crop = corrupt(geometry(image), *lookup[condition])
    tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(wt.mean, wt.std)])(crop)
    return crop, tensor


def predict_image(image, predictor, condition="clean"):
    model, device, policy = predictor
    view = policy.get("preprocessing", "center_crop")
    crop, tensor = prepare_image(image, condition, view)
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device))
        if not torch.isfinite(logits).all():
            raise ValueError("Model returned invalid scores.")
        probabilities = (logits / policy["temperature"]).softmax(1)[0].cpu()
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    accepted = confidence >= policy["threshold"]
    result = dict(category=CLASSES[index], confirmed_category=CLASSES[index] if accepted else None,
                  preprocessing=view, calibrated_confidence=confidence,
                  decision="Accepted" if accepted else "Manual review needed",
                  threshold=policy["threshold"], condition=condition,
                  class_scores={name: float(probabilities[i]) for i, name in enumerate(CLASSES)},
                  model_sha256=policy["checkpoint_sha256"],
                  note="Confidence is an estimate, not a guarantee. One primary item among five known materials.")
    return crop, result

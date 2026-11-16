"""Geometry shared by calibration, evaluation and app inference."""
from PIL import ImageOps, Image
from torchvision import transforms
from torchvision.models import ResNet18_Weights


class FullImage:
    def __call__(self, image):
        # Preserve cap, base and proportions; letterbox rather than center-crop.
        return ImageOps.pad(image, (224, 224), method=Image.Resampling.BILINEAR,
                            color=(255, 255, 255), centering=(0.5, 0.5))


def geometry_for(view):
    if view == "full_frame":
        return FullImage()
    if view != "center_crop":
        raise ValueError(f"Unknown preprocessing: {view}")
    wt = ResNet18_Weights.DEFAULT.transforms()
    return transforms.Compose([
        transforms.Resize(wt.resize_size, interpolation=wt.interpolation, antialias=wt.antialias),
        transforms.CenterCrop(wt.crop_size)])

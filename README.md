# SmartSort AI

**Waste classification under challenging conditions.** A PyTorch image classifier and Streamlit demo for cardboard, glass, metal, paper and plastic. The project measures how blur, low light and partial occlusion affect predictions, and flags low-confidence results for manual review.

## Features

- Pretrained ResNet18 fine-tuning: clean baseline and a separate augmented model.
- Dataset integrity checks, conflicting-duplicate exclusion and grouped train/validation/test splits.
- Clean-test metrics, confusion matrices and synthetic robustness experiments.
- Temperature scaling and a validation-tuned review threshold, evaluated separately on test images.
- Local image upload, dataset examples, quality simulations and downloadable predictions.
- Optional whole-photo padding preserves object ends; it uses a separate calibration policy.

## Setup (Windows PowerShell)

The development machine used Python 3.14.3, an NVIDIA RTX 4050 Laptop GPU (6 GB) and CUDA-enabled PyTorch. Training currently requires an NVIDIA GPU; inference can fall back to CPU. Package versions below reflect the tested environment.

```powershell
git clone https://github.com/Priyans123657/SmartSortAI.git
cd SmartSortAI
python -m venv .venv
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe check_setup.py
```

These commands use the environment's interpreter directly; activation is optional. A compatible NVIDIA driver is required. If using another platform, choose an appropriate build from the [PyTorch installer](https://pytorch.org/get-started/locally/).

## Dataset

Download `dataset-resized.zip` from [TrashNet by Gary Thung and Mindy Yang](https://github.com/garythung/trashnet), also available through the [author's Hugging Face dataset](https://huggingface.co/datasets/garythung/trashnet/tree/main). Respect the dataset's own license and attribution. Images are not redistributed in this repository.

Extract so the folders are directly inside:

```text
data/dataset-resized/
  cardboard/
  glass/
  metal/
  paper/
  plastic/
  trash/
```

The `trash` category is excluded from the five-class model. Do not combine original and resized versions across splits: they represent the same underlying images.

```powershell
.\.venv\Scripts\python.exe check_dataset.py
.\.venv\Scripts\python.exe prepare_dataset.py
```

Preparation preserves source images and writes manifests under `data/splits/`. In the recorded run, 2,384 usable images were split into 1,668 training, 359 validation and 357 test images. Similarity grouping is a heuristic and does not guarantee every related photograph is detected.

## Train and evaluate

```powershell
.\.venv\Scripts\python.exe train.py
.\.venv\Scripts\python.exe train.py --augmented
```

Each run writes its own timestamped directory under `models/`. The first real training run downloads pretrained ResNet18 weights. Replace `YOUR_RUN_DIRECTORY` below with the directory printed by training:

```powershell
$checkpoint = '.\models\YOUR_RUN_DIRECTORY\best.pt'
.\.venv\Scripts\python.exe evaluate.py --checkpoint $checkpoint
.\.venv\Scripts\python.exe evaluate_robustness.py --checkpoint $checkpoint
.\.venv\Scripts\python.exe calibrate.py --checkpoint $checkpoint
.\.venv\Scripts\python.exe evaluate_uncertainty.py --checkpoint $checkpoint
```

HTML and CSV reports are saved under `results/`. Calibration fits confidence and review settings using validation data, without loading test images. Validation was also used for checkpoint selection; calibration tuning accuracy is not an independent performance estimate. Keep test results for reporting rather than repeatedly tuning against them.

Optional whole-photo mode:

```powershell
.\.venv\Scripts\python.exe calibrate.py --checkpoint $checkpoint --full-frame
.\.venv\Scripts\python.exe evaluate_uncertainty.py --checkpoint $checkpoint --full-frame
```

## Run the app

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL printed by Streamlit and select a saved model. A matching calibration JSON is required for calibrated predictions. Stop the server with **Ctrl+C** in the terminal.

**Model weights, datasets, local reports and personal test photographs are not included.** Train and calibrate locally before using the demo. Uploading this repository does not deploy a hosted application.

## Recorded results

Accuracy on the same 357 held-out TrashNet images, with synthetic conditions applied for stress tests:

| Condition | Baseline | Augmented |
|---|---:|---:|
| Clean | 91.32% | 87.96% |
| Strong blur | 61.06% | 73.11% |
| Strong low light | 34.17% | 84.03% |
| Strong occlusion | 67.23% | 80.67% |

Augmentation improved these strong-corruption results while reducing clean-image accuracy. These are recorded experimental results, not guaranteed outcomes for new photographs or reruns. Synthetic occlusion is not a demonstration of multi-object detection.

## Limitations

- Designed for one primary object and five known material classes; no instance segmentation, conveyor control or contamination detection.
- Real-world photos can differ substantially from TrashNet. Plastic bottles have been confused with glass or metal in manual trials.
- A confidence score is not a guarantee. Below the calibrated threshold the app shows **Uncertain material** and a tentative guess. Accepted predictions can also be wrong.
- Whole-photo padding fixes geometric cropping but does not guarantee better classification. Its recorded clean accuracy was lower than the original center-crop pipeline, so it remains experimental.
- New real-world data should be labeled and split by physical object/source before additional training. Keep a separate unseen evaluation set.

## Code map

| File | Purpose |
|---|---|
| `check_setup.py`, `check_dataset.py` | Environment and image checks |
| `prepare_dataset.py` | Duplicate screening and split manifests |
| `train.py` | Baseline and augmented training |
| `evaluate.py`, `evaluate_robustness.py` | Clean and stress-test reports |
| `calibrate.py`, `evaluate_uncertainty.py` | Confidence calibration and review-policy evaluation |
| `preprocessing.py`, `inference.py` | Shared image geometry and prediction |
| `app.py` | Streamlit demo |

Built with Python, PyTorch, torchvision, Pillow and Streamlit.

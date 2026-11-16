r"""Run this first: .\.venv\Scripts\python.exe check_setup.py"""
import importlib.util
import platform
import sys

print("SmartSort AI - setup check")
print("Python:", platform.python_version())
print("Interpreter:", sys.executable)
print("Categories: cardboard, glass, metal, paper, plastic")

if importlib.util.find_spec("torch") is None:
    print("NEXT STEP: Install PyTorch. Model training is not ready yet.")
else:
    import torch
    print("PyTorch:", torch.__version__)
    print("GPU available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        x = torch.ones((32, 32), device="cuda")
        result = x @ x
        torch.cuda.synchronize()
        assert result[0, 0].item() == 32
        print("GPU calculation: PASS")
    else:
        print("NEXT STEP: Diagnose GPU support before training.")

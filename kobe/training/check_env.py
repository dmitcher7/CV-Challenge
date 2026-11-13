"""Environment check for the smoke test.

Prints the Python, ultralytics and torch versions and whether a CUDA GPU
is available. Missing packages are reported instead of crashing.
"""

import sys

print(f"Python      : {sys.version.split()[0]}  ({sys.executable})")

# ultralytics is the YOLO library we train with
try:
    import ultralytics
    print(f"ultralytics : {ultralytics.__version__}")
except ImportError:
    print("ultralytics : NOT INSTALLED")

# torch is the deep learning backend used by ultralytics
try:
    import torch
    print(f"torch       : {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"CUDA        : {cuda}")
    # Only ask for a GPU name when CUDA is actually available
    print(f"GPU         : {torch.cuda.get_device_name(0) if cuda else 'none (training will run on CPU)'}")
except ImportError:
    print("torch       : NOT INSTALLED")

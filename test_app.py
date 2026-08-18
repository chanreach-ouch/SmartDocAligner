import numpy as np
import sys
import os
# add current dir to path
sys.path.append(os.path.dirname(__file__))
from app import process_image

img = np.zeros((100, 100, 3), dtype=np.uint8)

try:
    res = process_image(img, "OpenCV crop method")
    print("OpenCV works", len(res))
except Exception as e:
    print(f"OpenCV failed: {e}")

try:
    res = process_image(img, "Document Segmentation (DALAI)")
    print("DALAI works", len(res))
except Exception as e:
    print(f"DALAI failed: {e}")

import cv2
import numpy as np


# Blur image to reduce camera noise
def reduce_noise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    return cv2.GaussianBlur(image, (k, k), 0)


# Boost contrast using CLAHE without altering colors
def adaptive_contrast(image: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


# Run the full preprocessing pipeline (denoise + contrast)
def preprocess_pipeline(image: np.ndarray, denoise: bool = True, enhance_contrast: bool = True) -> np.ndarray:
    out = image.copy()
    if denoise:
        out = reduce_noise(out)
    if enhance_contrast:
        out = adaptive_contrast(out)
    return out

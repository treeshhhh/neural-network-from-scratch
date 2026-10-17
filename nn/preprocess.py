"""Deterministic MNIST-style preprocessing using NumPy only."""
import numpy as np


def bounding_box(image: np.ndarray, threshold: float = 8 / 255):
    """Return exclusive (top, left, bottom, right), or None for a blank image."""
    rows, cols = np.nonzero(image > threshold)
    if not len(rows):
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()) + 1, int(cols.max()) + 1


def _weights(source: int, target: int) -> np.ndarray:
    if source >= target:
        # Exact pixel-area overlap: antialias before downsampling.
        start = np.arange(target, dtype=np.float64)[:, None] * source / target
        end = (np.arange(target, dtype=np.float64)[:, None] + 1) * source / target
        pixels = np.arange(source, dtype=np.float64)[None, :]
        return np.maximum(0, np.minimum(end, pixels + 1) - np.maximum(start, pixels)) / (source / target)
    # Bilinear pixel-center interpolation for small drawings being enlarged.
    positions = np.clip((np.arange(target) + .5) * source / target - .5, 0, source - 1)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, source - 1)
    fraction = positions - left
    weights = np.zeros((target, source), dtype=np.float64)
    weights[np.arange(target), left] += 1 - fraction
    weights[np.arange(target), right] += fraction
    return weights


def resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    return (_weights(image.shape[0], height) @ image @ _weights(image.shape[1], width).T).astype(np.float32)


def preprocess(pixels: np.ndarray):
    """Accept a finite grayscale [0,255] image, return normalized (1,784) + info.

    Blank inputs return zeros and blank=True; callers should abstain rather than
    showing a confident prediction for a blank canvas.
    """
    raw = np.asarray(pixels)
    if raw.ndim != 2 or not all(1 <= n <= 280 for n in raw.shape) or raw.dtype.kind not in 'iuf':
        raise ValueError('expected a numeric grayscale image, dimensions 1..280')
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 255):
        raise ValueError('pixels must be finite values in [0,255]')
    image = raw.astype(np.float32) / 255.0
    bbox = bounding_box(image)
    result = np.zeros((28, 28), dtype=np.float32)
    if bbox is None:
        return result.reshape(1, 784), {'blank': True, 'bbox': None, 'resized_shape': None, 'shift': [0, 0]}
    top, left, bottom, right = bbox
    crop = image[top:bottom, left:right]
    scale = 20 / max(crop.shape)
    h, w = [max(1, min(20, int(round(n * scale)))) for n in crop.shape]
    digit = resize(crop, h, w)
    y0, x0 = (28 - h) // 2, (28 - w) // 2
    result[y0:y0 + h, x0:x0 + w] = digit
    mass = float(result.sum())
    yy, xx = np.indices((28, 28))
    center_y, center_x = float((result * yy).sum() / mass), float((result * xx).sum() / mass)
    # Integer translation preserves intensities; clamp to keep all ink on canvas.
    dy = int(np.clip(round(13.5 - center_y), -y0, 28 - y0 - h))
    dx = int(np.clip(round(13.5 - center_x), -x0, 28 - x0 - w))
    shifted = np.zeros_like(result)
    shifted[y0 + dy:y0 + dy + h, x0 + dx:x0 + dx + w] = digit
    return shifted.reshape(1, 784), {'blank': False, 'bbox': list(bbox),
                                    'resized_shape': [h, w], 'shift': [dy, dx]}

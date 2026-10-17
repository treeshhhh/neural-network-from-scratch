"""Vectorized dense classifier: NumPy only, no automatic differentiation."""
from pathlib import Path
from typing import Sequence
import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits)
    if z.ndim != 2 or z.shape[0] == 0 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('logits must be a finite, nonempty (batch, classes) matrix')
    shifted = z - z.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    # Log-sum-exp avoids log(softmax) underflow, even for a confident wrong class.
    z = np.asarray(logits)
    if z.ndim != 2 or not len(z) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('invalid logits')
    y = np.asarray(labels)
    if y.shape != (len(z),) or y.dtype.kind not in 'iu' or np.any(y < 0) or np.any(y >= z.shape[1]):
        raise ValueError('labels must be integer class indices, one per example')
    shifted = z - z.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y]))


class MLP:
    def __init__(self, sizes: Sequence[int] = (784, 128, 64, 10), seed: int = 42,
                 dtype: str = 'float32') -> None:
        if len(sizes) < 2 or any(type(s) is not int or s < 1 for s in sizes) or sizes[-1] < 2:
            raise ValueError('sizes must contain positive integer dimensions and at least two classes')
        if dtype not in ('float32', 'float64'):
            raise ValueError('dtype must be float32 or float64')
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.weights = []
        self.biases = []
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            # He for ReLU hidden layers; Xavier-like variance for linear logits.
            scale = np.sqrt((2.0 if i < len(sizes) - 2 else 1.0) / fan_in)
            self.weights.append((rng.standard_normal((fan_in, fan_out)) * scale).astype(dtype))
            self.biases.append(np.zeros(fan_out, dtype=dtype))

    def _input(self, x: np.ndarray) -> np.ndarray:
        a = np.asarray(x, dtype=self.weights[0].dtype)
        if a.ndim != 2 or not len(a) or a.shape[1] != self.sizes[0] or not np.isfinite(a).all():
            raise ValueError('input must be finite (nonempty batch, input dimension)')
        return a

    def _forward(self, x: np.ndarray):
        activations = [self._input(x)]
        preactivations = []
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = activations[-1] @ w + b
            preactivations.append(z)
            activations.append(np.maximum(z, 0) if i < len(self.weights) - 1 else z)
        return activations, preactivations

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward(x)[0][-1]

    def loss_and_gradients(self, x: np.ndarray, y: np.ndarray):
        a, z = self._forward(x)
        loss = cross_entropy(a[-1], y)
        delta = softmax(a[-1])
        delta[np.arange(len(y)), y] -= 1
        delta /= len(y)  # Mean loss: normalize exactly once.
        dw, db = [None] * len(self.weights), [None] * len(self.weights)
        for i in range(len(self.weights) - 1, -1, -1):
            dw[i] = a[i].T @ delta
            db[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.weights[i].T) * (z[i - 1] > 0)
        return loss, dw, db

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return softmax(self.forward(x))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x).argmax(axis=1)

    def save(self, path: str) -> None:
        path = Path(path)
        if path.suffix != '.npz':
            raise ValueError('checkpoint path must end in .npz')
        path.parent.mkdir(parents=True, exist_ok=True)
        params = {'sizes': np.asarray(self.sizes, dtype=np.int64)}
        params.update({f'w{i}': w for i, w in enumerate(self.weights)})
        params.update({f'b{i}': b for i, b in enumerate(self.biases)})
        np.savez_compressed(path, **params)

    @classmethod
    def load(cls, path: str) -> 'MLP':
        with np.load(path, allow_pickle=False) as saved:
            dims = saved['sizes']
            if dims.ndim != 1 or dims.dtype.kind not in 'iu':
                raise ValueError('invalid checkpoint architecture')
            # Reject unreasonable allocation sizes before initialization.
            sizes = tuple(int(s) for s in dims)
            if len(sizes) > 20 or any(s < 1 or s > 10000 for s in sizes):
                raise ValueError('invalid checkpoint dimensions')
            if sum(a * b for a, b in zip(sizes[:-1], sizes[1:])) > 10_000_000:
                raise ValueError('checkpoint too large')
            dtype = str(saved['w0'].dtype)
            model = cls(sizes, dtype=dtype)
            expected = {'sizes'} | {f'{kind}{i}' for kind in ('w', 'b') for i in range(len(model.weights))}
            if set(saved.files) != expected:
                raise ValueError('checkpoint keys do not match architecture')
            for i in range(len(model.weights)):
                w, b = saved[f'w{i}'], saved[f'b{i}']
                if w.shape != model.weights[i].shape or b.shape != model.biases[i].shape:
                    raise ValueError('checkpoint parameter shape mismatch')
                if w.dtype != model.weights[i].dtype or b.dtype != w.dtype or not np.isfinite(w).all() or not np.isfinite(b).all():
                    raise ValueError('invalid checkpoint parameters')
                model.weights[i], model.biases[i] = w.copy(), b.copy()
        return model


class SGD:
    """Classical momentum: v <- mu*v + grad; parameter <- parameter - lr*v."""
    def __init__(self, model: MLP, lr: float = 0.1, momentum: float = 0.9) -> None:
        if not np.isfinite(lr) or lr <= 0 or not np.isfinite(momentum) or not 0 <= momentum < 1:
            raise ValueError('lr must be positive and momentum in [0, 1)')
        self.lr, self.momentum = lr, momentum
        self.vw = [np.zeros_like(w) for w in model.weights]
        self.vb = [np.zeros_like(b) for b in model.biases]

    def step(self, model: MLP, dw, db) -> None:
        if len(dw) != len(model.weights) or len(db) != len(model.biases):
            raise ValueError('incorrect gradient count')
        for w, b, gw, gb in zip(model.weights, model.biases, dw, db):
            if gw.shape != w.shape or gb.shape != b.shape or not np.isfinite(gw).all() or not np.isfinite(gb).all():
                raise ValueError('invalid gradients')
        for i in range(len(model.weights)):
            self.vw[i] *= self.momentum
            self.vb[i] *= self.momentum
            self.vw[i] += dw[i]
            self.vb[i] += db[i]
            model.weights[i] -= self.lr * self.vw[i]
            model.biases[i] -= self.lr * self.vb[i]

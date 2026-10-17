"""MNIST IDX loading, integrity checking, and disjoint stratified splits."""
import gzip
import hashlib
import os
from pathlib import Path
import struct
import urllib.request
import numpy as np

BASE = 'https://storage.googleapis.com/cvdf-datasets/mnist/'
FILES = {
    'train-images-idx3-ubyte.gz': 'f68b3c2dcbeaaa9fbdd348bbdeb94873',
    'train-labels-idx1-ubyte.gz': 'd53e105ee54ea40749a09fcbcd1e9432',
    't10k-images-idx3-ubyte.gz': '9fb629c4189551a2d022fa330f9573f3',
    't10k-labels-idx1-ubyte.gz': 'ec29112dd5afa0611ce80d1b7f02629c',
}


def read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, 'rb') as f:
        raw = f.read()
    if len(raw) < 8:
        raise ValueError('truncated IDX header')
    magic, count = struct.unpack('>II', raw[:8])
    if magic == 2051:
        if len(raw) < 16:
            raise ValueError('truncated image header')
        rows, cols = struct.unpack('>II', raw[8:16])
        if rows != 28 or cols != 28:
            raise ValueError('expected 28x28 images')
        offset, shape = 16, (count, 784)
    elif magic == 2049:
        offset, shape = 8, (count,)
    else:
        raise ValueError('unsupported IDX magic')
    expected = count * (784 if magic == 2051 else 1)
    if not count or len(raw) - offset != expected:
        raise ValueError('incorrect IDX payload length')
    result = np.frombuffer(raw, dtype=np.uint8, offset=offset).reshape(shape).copy()
    if magic == 2049 and np.any(result > 9):
        raise ValueError('invalid MNIST label')
    return result


def load_mnist(directory: str):
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name, digest in FILES.items():
        path = root / name
        if not path.exists():
            tmp = path.with_suffix('.download')
            try:
                with urllib.request.urlopen(BASE + name, timeout=60) as response:
                    content = response.read()
                if hashlib.md5(content).hexdigest() != digest:
                    raise ValueError('MNIST download integrity mismatch: ' + name)
                tmp.write_bytes(content)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        if hashlib.md5(path.read_bytes()).hexdigest() != digest:
            raise ValueError('MNIST file integrity mismatch; remove and download again: ' + str(path))
        arrays.append(read_idx(path))
    x, y, xt, yt = arrays
    if len(x) != 60000 or len(y) != 60000 or len(xt) != 10000 or len(yt) != 10000:
        raise ValueError('unexpected MNIST dataset sizes')
    return x.astype(np.float32) / 255.0, y.astype(np.int64), xt.astype(np.float32) / 255.0, yt.astype(np.int64)


def split_indices(labels: np.ndarray, seed: int = 42, validation_size: int = 10000):
    if labels.ndim != 1 or labels.dtype.kind not in 'iu' or np.any(labels < 0) or np.any(labels > 9):
        raise ValueError('expected MNIST class indices')
    if type(validation_size) is not int or validation_size < 10 or validation_size >= len(labels) or validation_size % 10:
        raise ValueError('validation size must be a multiple of 10, at least 10, smaller than dataset')
    rng = np.random.default_rng(seed)
    train, val = [], []
    per_class = validation_size // 10
    for digit in range(10):
        indices = rng.permutation(np.flatnonzero(labels == digit))
        if len(indices) <= per_class:
            raise ValueError('too few examples per class')
        val.extend(indices[:per_class])
        train.extend(indices[per_class:])
    return rng.permutation(train), rng.permutation(val)


def epoch_order(size: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)

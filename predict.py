#!/usr/bin/env python3
"""Evaluate a saved checkpoint or predict on normalized .npy rows."""
import argparse
import json
import numpy as np
from nn.model import MLP
from nn.data import load_mnist
from nn.evaluate import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/numpy_best.npz')
    parser.add_argument('--input', help='.npy float array of normalized pixels, shape (N,784)')
    parser.add_argument('--data-dir', default='data/mnist')
    args = parser.parse_args()
    model = MLP.load(args.model)
    if args.input:
        images = np.load(args.input, allow_pickle=False)
        if not np.isfinite(images).all() or np.any(images < 0) or np.any(images > 1):
            parser.error('input pixels must be normalized to [0,1]')
        print(json.dumps({'predictions': model.predict(images).tolist()}))
    else:
        _, _, images, labels = load_mnist(args.data_dir)
        metrics, _ = evaluate(model, images, labels)
        print(json.dumps(metrics, indent=2))

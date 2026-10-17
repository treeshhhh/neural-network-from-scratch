#!/usr/bin/env python3
"""One bounded MNIST experiment; optional matched PyTorch comparison."""
import os
# Set thread limits before importing NumPy/BLAS.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
import argparse
import json
from pathlib import Path
import platform
import time
import numpy as np
from nn.data import load_mnist, split_indices, epoch_order
from nn.model import MLP, SGD
from nn.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/mnist')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--torch', action='store_true', help='run the optional matched CPU baseline')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.seed < 0:
        parser.error('epochs/batch-size must be positive; seed must be nonnegative')
    config = vars(args).copy()
    config['architecture'] = [784, 128, 64, 10]
    initial = MLP(config['architecture'], seed=args.seed)
    model = MLP(config['architecture'], seed=args.seed)
    optimizer = SGD(model, args.lr, args.momentum)
    xall, yall, xt, yt = load_mnist(args.data_dir)
    train_idx, val_idx = split_indices(yall, args.seed)
    x, y, xv, yv = xall[train_idx], yall[train_idx], xall[val_idx], yall[val_idx]
    Path('results').mkdir(exist_ok=True)
    np.savez_compressed('results/split_indices.npz', train=train_idx, validation=val_idx)
    history, best_loss, selected_epoch = [], float('inf'), 0
    best_parameters = None
    start = time.perf_counter()
    for epoch in range(args.epochs):
        order = epoch_order(len(y), args.seed, epoch)
        for offset in range(0, len(y), args.batch_size):
            indices = order[offset:offset + args.batch_size]
            _, dw, db = model.loss_and_gradients(x[indices], y[indices])
            optimizer.step(model, dw, db)
        train_metrics, _ = evaluate(model, x, y)
        val_metrics, _ = evaluate(model, xv, yv)
        history.append({'epoch': epoch + 1, 'train_loss': train_metrics['loss'],
                        'train_accuracy': train_metrics['accuracy'], 'validation_loss': val_metrics['loss'],
                        'validation_accuracy': val_metrics['accuracy']})
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            selected_epoch = epoch + 1
            best_parameters = ([w.copy() for w in model.weights], [b.copy() for b in model.biases])
        print('numpy epoch %d: train %.4f val %.4f' % (epoch + 1, train_metrics['accuracy'], val_metrics['accuracy']), flush=True)
    seconds = time.perf_counter() - start
    model.weights, model.biases = best_parameters
    model.save('checkpoints/numpy_best.npz')
    train_metrics, _ = evaluate(model, x, y)
    val_metrics, _ = evaluate(model, xv, yv)
    # The test set is evaluated only after validation-based checkpoint selection.
    test_metrics, predictions = evaluate(model, xt, yt)
    result = {'config': config, 'environment': {'python': platform.python_version(),
              'numpy': np.__version__, 'platform': platform.platform(), 'thread_limits_requested': 1,
              'numpy_blas': np.__config__.CONFIG.get('Build Dependencies', {}).get('blas', {}).get('name', 'unknown')},
              'counts': {'train': len(y), 'validation': len(yv), 'test': len(yt)},
              'numpy': {'framework': 'NumPy', 'training_seconds': seconds, 'history': history,
                        'selected_epoch': selected_epoch, 'train': train_metrics,
                        'validation': val_metrics, 'test': test_metrics}}
    if args.torch:
        try:
            from baseline import train_torch
            result['pytorch'], _ = train_torch(initial, x, y, xv, yv, xt, yt, config)
        except ImportError as exc:
            result['pytorch'] = {'status': 'unavailable', 'reason': str(exc)}
    else:
        result['pytorch'] = {'status': 'not requested'}
    result['timing_notes'] = ('perf_counter elapsed wall time; includes training, epoch train/validation evaluation, and in-memory best-parameter copies; excludes data loading, setup, final test, disk checkpoint writes, and plots. One sequential run, NumPy first; actual NumPy BLAS concurrency not independently verified.')
    if 'test' in result['pytorch']:
        result['environment']['torch_threads'] = 1
    # Small fixed sample: original test order, not selected for correct answers.
    np.savez_compressed('results/examples.npz', images=xt[:36].reshape(-1, 28, 28),
                        labels=yt[:36], predictions=predictions[:36])
    Path('results/metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'numpy_test': test_metrics['accuracy'], 'numpy_seconds': seconds,
                      'pytorch_test': result['pytorch'].get('test', {}).get('accuracy')}, indent=2))


if __name__ == '__main__':
    main()

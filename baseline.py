"""Optional PyTorch reference. NumPy implementation never imports this module."""
import time
import numpy as np
from nn.data import epoch_order
from nn.model import MLP


def train_torch(initial: MLP, x, y, xv, yv, xt, yt, config):
    import torch
    from torch import nn
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    torch.use_deterministic_algorithms(True)
    layers = []
    for i, (a, b) in enumerate(zip(initial.sizes[:-1], initial.sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(initial.weights) - 1:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers).cpu()
    dense = [layer for layer in model if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for i, layer in enumerate(dense):
            layer.weight.copy_(torch.from_numpy(initial.weights[i].T.copy()))
            layer.bias.copy_(torch.from_numpy(initial.biases[i].copy()))
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=config['momentum'])
    criterion = nn.CrossEntropyLoss()
    tx, ty = torch.from_numpy(x), torch.from_numpy(y)
    tvx, tvy = torch.from_numpy(xv), torch.from_numpy(yv)

    def metrics(data, labels):
        model.eval()
        loss, correct = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(labels), 1024):
                logits = model(data[start:start + 1024])
                target = labels[start:start + 1024]
                loss += criterion(logits, target).item() * len(target)
                correct += (logits.argmax(1) == target).sum().item()
        return loss / len(labels), correct / len(labels)

    history, best_loss, best = [], float('inf'), None
    start_time = time.perf_counter()
    for epoch in range(config['epochs']):
        model.train()
        for offset in range(0, len(y), config['batch_size']):
            # Compute the same permutation once per epoch (outside this loop below).
            if offset == 0:
                order = epoch_order(len(y), config['seed'], epoch)
            indices = torch.from_numpy(order[offset:offset + config['batch_size']])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(tx[indices]), ty[indices])
            loss.backward()
            optimizer.step()
        train_loss, train_accuracy = metrics(tx, ty)
        val_loss, val_accuracy = metrics(tvx, tvy)
        history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'train_accuracy': train_accuracy,
                        'validation_loss': val_loss, 'validation_accuracy': val_accuracy})
        if val_loss < best_loss:
            best_loss = val_loss
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        print('torch epoch %d: train %.4f val %.4f' % (epoch + 1, train_accuracy, val_accuracy), flush=True)
    seconds = time.perf_counter() - start_time
    model.load_state_dict(best)
    # Export into the common non-pickle NPZ format; no torch.load is required.
    exported = MLP(initial.sizes, seed=config['seed'])
    for i, layer in enumerate(dense):
        exported.weights[i] = layer.weight.detach().numpy().T.copy()
        exported.biases[i] = layer.bias.detach().numpy().copy()
    from nn.evaluate import evaluate
    train_metrics, _ = evaluate(exported, x, y)
    val_metrics, _ = evaluate(exported, xv, yv)
    test_metrics, predictions = evaluate(exported, xt, yt)
    exported.save('checkpoints/pytorch_best.npz')
    return {'framework': 'PyTorch', 'version': torch.__version__, 'training_seconds': seconds,
            'history': history, 'selected_epoch': min(history, key=lambda r: r['validation_loss'])['epoch'],
            'train': train_metrics, 'validation': val_metrics, 'test': test_metrics}, predictions

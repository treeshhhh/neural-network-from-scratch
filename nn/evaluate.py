import numpy as np
from .model import MLP, cross_entropy


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray, batch_size: int = 1024):
    if len(x) != len(y) or not len(y) or batch_size < 1:
        raise ValueError('evaluation requires matching nonempty examples/labels and positive batch size')
    total_loss, predictions = 0.0, []
    for start in range(0, len(y), batch_size):
        logits = model.forward(x[start:start + batch_size])
        total_loss += len(logits) * cross_entropy(logits, y[start:start + batch_size])
        predictions.append(logits.argmax(axis=1))
    predictions = np.concatenate(predictions)
    matrix = np.zeros((model.sizes[-1], model.sizes[-1]), dtype=np.int64)
    np.add.at(matrix, (y, predictions), 1)
    return {'loss': total_loss / len(y), 'accuracy': float(np.mean(predictions == y)),
            'confusion_matrix': matrix.tolist()}, predictions

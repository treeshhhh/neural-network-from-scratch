"""Optional independent autograd reference; never used by nn/model.py."""
import importlib.util
import unittest
import numpy as np
from nn.model import MLP, SGD


@unittest.skipUnless(importlib.util.find_spec('torch'), 'optional torch not installed')
class TorchParityTests(unittest.TestCase):
    def test_forward_gradients_and_momentum_steps(self):
        import torch
        torch.set_num_threads(1)
        model = MLP((3, 4, 2), seed=8, dtype='float64')
        net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.ReLU(), torch.nn.Linear(4, 2)).double()
        dense = [net[0], net[2]]
        with torch.no_grad():
            for i, layer in enumerate(dense):
                layer.weight.copy_(torch.from_numpy(model.weights[i].T))
                layer.bias.copy_(torch.from_numpy(model.biases[i]))
        x = np.random.default_rng(9).normal(size=(5, 3)); y = np.array([0, 1, 0, 0, 1])
        opt = SGD(model, .03, .8)
        reference = torch.optim.SGD(net.parameters(), lr=.03, momentum=.8)
        for _ in range(2):
            logits = net(torch.from_numpy(x))
            np.testing.assert_allclose(model.forward(x), logits.detach().numpy(), atol=1e-12)
            loss, dw, db = model.loss_and_gradients(x, y)
            reference.zero_grad()
            torch_loss = torch.nn.functional.cross_entropy(logits, torch.from_numpy(y))
            self.assertAlmostEqual(loss, torch_loss.item(), places=12)
            torch_loss.backward()
            for i, layer in enumerate(dense):
                np.testing.assert_allclose(dw[i], layer.weight.grad.numpy().T, atol=1e-12)
                np.testing.assert_allclose(db[i], layer.bias.grad.numpy(), atol=1e-12)
            opt.step(model, dw, db); reference.step()
            for i, layer in enumerate(dense):
                np.testing.assert_allclose(model.weights[i], layer.weight.detach().numpy().T, atol=1e-12)
                np.testing.assert_allclose(model.biases[i], layer.bias.detach().numpy(), atol=1e-12)

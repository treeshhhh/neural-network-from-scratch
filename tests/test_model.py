import gzip
import math
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from nn.model import MLP, SGD, softmax, cross_entropy
from nn.evaluate import evaluate
from nn.data import read_idx, split_indices, epoch_order


class ModelTests(unittest.TestCase):
    def test_forward_shapes(self):
        m = MLP((4, 5, 3), seed=1)
        self.assertEqual(m.forward(np.ones((7, 4))).shape, (7, 3))
        self.assertEqual(m.predict(np.ones((7, 4))).shape, (7,))

    def test_softmax_normalization(self):
        p = softmax(np.array([[1., 2., 3.], [-2., 0., 9.]]))
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-14)
        self.assertTrue(np.all(p >= 0))
        np.testing.assert_allclose(p, softmax(np.array([[1., 2., 3.], [-2., 0., 9.]]) + 500))

    def test_stable_loss(self):
        logits = np.array([[1000., -1000., 0.], [-1000., 1000., 0.]])
        self.assertTrue(np.isfinite(softmax(logits)).all())
        self.assertEqual(cross_entropy(logits, np.array([1, 0])), 2000.)
        self.assertAlmostEqual(cross_entropy(np.zeros((2, 3)), np.array([1, 2])), math.log(3))
        self.assertAlmostEqual(cross_entropy(np.log([[.1, .7, .2]]), np.array([1])), -math.log(.7))

    def test_all_gradients_finite_differences(self):
        model = MLP((3, 4, 3, 2), seed=7, dtype='float64')
        x = np.random.default_rng(17).normal(size=(5, 3))
        y = np.array([0, 1, 1, 0, 1])
        _, z = model._forward(x)
        self.assertGreater(min(np.min(np.abs(a)) for a in z[:-1]), 1e-5)
        _, dw, db = model.loss_and_gradients(x, y)
        errors = []
        for parameter, gradient in zip(model.weights + model.biases, dw + db):
            for idx in np.ndindex(parameter.shape):
                old = parameter[idx]
                parameter[idx] = old + 1e-5
                plus = cross_entropy(model.forward(x), y)
                parameter[idx] = old - 1e-5
                minus = cross_entropy(model.forward(x), y)
                parameter[idx] = old
                numeric = (plus - minus) / 2e-5
                errors.append(abs(numeric - gradient[idx]))
                self.assertAlmostEqual(numeric, gradient[idx], delta=1e-7)
        print('Finite differences: %d parameters; max absolute error %.3g' % (len(errors), max(errors)))

    def test_relu_zero_derivative(self):
        model = MLP((2, 2, 2), dtype='float64')
        model.weights[0][:] = 0
        model.biases[0][:] = -1
        _, dw, db = model.loss_and_gradients(np.ones((3, 2)), np.array([0, 1, 0]))
        np.testing.assert_array_equal(dw[0], 0)
        np.testing.assert_array_equal(db[0], 0)
        np.testing.assert_array_equal(dw[1], 0)

    def test_sgd_and_momentum_updates(self):
        for momentum in (0., .9):
            model = MLP((2, 2), dtype='float64')
            old = model.weights[0].copy()
            optimizer = SGD(model, lr=.1, momentum=momentum)
            dw, db = [np.ones_like(old)], [np.ones(2)]
            optimizer.step(model, dw, db)
            np.testing.assert_allclose(model.weights[0], old - .1)
            optimizer.step(model, dw, db)
            np.testing.assert_allclose(model.weights[0], old - .1 - .1 * (1 + momentum))

    def test_training_reduces_loss(self):
        model = MLP((2, 2), seed=2, dtype='float64')
        x = np.array([[1., 1.], [-1., -1.], [2., 1.], [-2., -1.]])
        y = np.array([0, 1, 0, 1])
        before = cross_entropy(model.forward(x), y)
        opt = SGD(model, .1, 0.)
        for _ in range(30):
            _, dw, db = model.loss_and_gradients(x, y)
            opt.step(model, dw, db)
        self.assertLess(cross_entropy(model.forward(x), y), before)
        np.testing.assert_array_equal(model.predict(x), y)

    def test_checkpoint_prediction_consistency(self):
        m = MLP((3, 5, 2), seed=12)
        x = np.random.default_rng(1).normal(size=(4, 3))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / 'model.npz')
            m.save(path)
            loaded = MLP.load(path)
            np.testing.assert_array_equal(loaded.predict_proba(x), m.predict_proba(x))
            np.testing.assert_array_equal(loaded.predict(x), m.predict(x))
            np.savez(path, sizes=np.array([3, 2]), w0=np.zeros((2, 2)), b0=np.zeros(2))
            with self.assertRaises(ValueError):
                MLP.load(path)

    def test_deterministic_training(self):
        x = np.ones((3, 2)); y = np.array([0, 1, 0])
        models = [MLP((2, 3, 2), seed=5) for _ in range(2)]
        for m in models:
            opt = SGD(m)
            for _ in range(3):
                _, dw, db = m.loss_and_gradients(x, y)
                opt.step(m, dw, db)
        for a, b in zip(models[0].weights, models[1].weights):
            np.testing.assert_array_equal(a, b)

    def test_evaluation_batch_invariance_and_no_mutation(self):
        m = MLP((3, 2), seed=1)
        x = np.ones((7, 3)); y = np.array([0, 1, 0, 1, 1, 0, 1])
        old = m.weights[0].copy()
        a, p = evaluate(m, x, y, 2); b, q = evaluate(m, x, y, 7)
        self.assertAlmostEqual(a['loss'], b['loss'], places=6)
        np.testing.assert_array_equal(p, q)
        self.assertEqual(np.sum(a['confusion_matrix']), 7)
        np.testing.assert_array_equal(m.weights[0], old)

    def test_input_errors(self):
        for sizes in ((2,), (2, 0, 2), (2, 1)):
            with self.assertRaises(ValueError):
                MLP(sizes)
        model = MLP((3, 2))
        for x in (np.ones((2, 4)), np.empty((0, 3)), np.full((2, 3), np.nan)):
            with self.assertRaises(ValueError):
                model.forward(x)
        for y in (np.array([-1, 0]), np.array([2, 0]), np.array([.1, .2])):
            with self.assertRaises(ValueError):
                model.loss_and_gradients(np.ones((2, 3)), y)
        with self.assertRaises(ValueError):
            SGD(model, lr=-1)


class DataTests(unittest.TestCase):
    def test_split_disjoint_and_reproducible(self):
        y = np.repeat(np.arange(10), 20)
        t, v = split_indices(y, 42, 50)
        self.assertEqual(len(set(t) & set(v)), 0)
        self.assertEqual(len(set(t) | set(v)), len(y))
        np.testing.assert_array_equal(np.bincount(y[v]), [5] * 10)
        t2, v2 = split_indices(y, 42, 50)
        np.testing.assert_array_equal(t, t2)
        np.testing.assert_array_equal(v, v2)
        np.testing.assert_array_equal(epoch_order(100, 42, 1), epoch_order(100, 42, 1))
        self.assertFalse(np.array_equal(epoch_order(100, 42, 1), epoch_order(100, 42, 2)))

    def test_idx_headers_and_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.gz'
            cases = [(struct.pack('>II', 2049, 2) + bytes([3, 9]), (2,)),
                     (struct.pack('>IIII', 2051, 1, 28, 28) + bytes(784), (1, 784))]
            for payload, shape in cases:
                with gzip.open(path, 'wb') as f:
                    f.write(payload)
                self.assertEqual(read_idx(path).shape, shape)
            for payload in (b'', struct.pack('>II', 2049, 2) + b'\x00',
                            struct.pack('>II', 2049, 1) + b'\xff'):
                with gzip.open(path, 'wb') as f:
                    f.write(payload)
                with self.assertRaises(ValueError):
                    read_idx(path)


if __name__ == '__main__':
    unittest.main()

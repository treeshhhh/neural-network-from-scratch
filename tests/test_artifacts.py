"""Check shipped artifacts without downloading data or retraining."""
import json
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from nn.model import MLP

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless((ROOT / 'results/metrics.json').exists(), 'no recorded experiment')
class ArtifactTests(unittest.TestCase):
    def test_split_indices_and_confusion_counts(self):
        r = json.loads((ROOT / 'results/metrics.json').read_text())
        with np.load(ROOT / 'results/split_indices.npz', allow_pickle=False) as split:
            t, v = split['train'], split['validation']
            self.assertEqual((len(t), len(v)), (50000, 10000))
            self.assertEqual(len(np.intersect1d(t, v)), 0)
            np.testing.assert_array_equal(np.sort(np.concatenate([t, v])), np.arange(60000))
        for key in ('numpy', 'pytorch'):
            if 'test' not in r[key]:
                continue
            cm = np.asarray(r[key]['test']['confusion_matrix'])
            self.assertEqual(cm.shape, (10, 10))
            self.assertEqual(cm.sum(), 10000)
            self.assertAlmostEqual(np.trace(cm) / cm.sum(), r[key]['test']['accuracy'])

    def test_checkpoint_matches_shipped_examples(self):
        with np.load(ROOT / 'results/examples.npz', allow_pickle=False) as examples:
            m = MLP.load(ROOT / 'checkpoints/numpy_best.npz')
            self.assertEqual(sum(w.size + b.size for w, b in zip(m.weights, m.biases)), 109386)
            p = m.predict(examples['images'].reshape(-1, 784))
            np.testing.assert_array_equal(p, examples['predictions'])

    def test_generated_svg_parses(self):
        for name in ('loss.svg', 'accuracy.svg', 'confusion-numpy.svg', 'predictions.svg'):
            self.assertEqual(ET.parse(ROOT / 'results' / name).getroot().tag, '{http://www.w3.org/2000/svg}svg')

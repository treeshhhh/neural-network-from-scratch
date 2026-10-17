import http.client
import json
from pathlib import Path
import threading
import unittest
from unittest.mock import patch
import numpy as np
from app import DrawingServer, Predictor, MAX_BODY
from nn.model import MLP
from nn.preprocess import preprocess, bounding_box, resize

ROOT = Path(__file__).resolve().parents[1]


def payload(image):
    return {'width': image.shape[1], 'height': image.shape[0], 'pixels': image.astype(int).ravel().tolist()}


def stroke():
    im = np.zeros((100, 100), dtype=np.uint8)
    im[15:85, 44:56] = 255
    return im


class PreprocessTests(unittest.TestCase):
    def test_blank(self):
        x, info = preprocess(np.zeros((280, 280), dtype=np.uint8))
        self.assertTrue(info['blank'])
        self.assertEqual(x.shape, (1, 784))
        self.assertEqual(x.dtype, np.float32)
        self.assertEqual(x.sum(), 0)
        _, info = preprocess(np.full((20, 20), 7, dtype=np.uint8))
        self.assertTrue(info['blank'])

    def test_bbox_aspect_and_normalization(self):
        im = stroke()
        self.assertEqual(bounding_box(im / 255), (15, 44, 85, 56))
        x, info = preprocess(im)
        self.assertEqual(info['bbox'], [15, 44, 85, 56])
        self.assertEqual(info['resized_shape'], [20, 3])
        self.assertEqual(x.shape, (1, 784))
        self.assertGreater(x.max(), 0)
        self.assertGreaterEqual(x.min(), 0)
        self.assertLessEqual(x.max(), 1)

    def test_mass_center_and_no_clipping(self):
        im = np.zeros((90, 110), dtype=np.uint8)
        im[4:70, 10:20] = 255
        im[55:70, 10:55] = 255
        x, info = preprocess(im)
        processed = x.reshape(28, 28)
        yy, xx = np.indices(processed.shape)
        self.assertLessEqual(abs((processed * yy).sum() / processed.sum() - 13.5), .51)
        self.assertLessEqual(abs((processed * xx).sum() / processed.sum() - 13.5), .51)
        t, l, b, r = info['bbox']; h, w = info['resized_shape']
        expected_mass = resize(im[t:b, l:r].astype(np.float32) / 255, h, w).sum()
        self.assertAlmostEqual(float(processed.sum()), float(expected_mass), places=4)

    def test_translation_invariance_and_determinism(self):
        a, b = np.zeros((100, 100)), np.zeros((100, 100))
        a[10:60, 10:20] = 255; b[30:80, 70:80] = 255
        x, _ = preprocess(a); y, _ = preprocess(b)
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, preprocess(a)[0])

    def test_antialias_area_average(self):
        checker = (np.indices((40, 40)).sum(axis=0) % 2).astype(np.float32)
        down = resize(checker, 20, 20)
        np.testing.assert_allclose(down, .5, atol=1e-7)
        up = resize(np.array([[0., 1.]], dtype=np.float32), 1, 4)
        np.testing.assert_allclose(up, [[0., .25, .75, 1.]])

    def test_row_major_orientation(self):
        image = np.zeros((28, 28), dtype=np.uint8)
        image[3:23, 5:8] = 255; image[20:23, 5:20] = 255
        x, _ = preprocess(image)
        grid = x.reshape(28, 28)
        self.assertGreater(grid[19:23, 15:20].sum(), grid[5:9, 15:20].sum())
        np.testing.assert_array_equal(x[0], grid.ravel(order='C'))

    def test_invalid_inputs(self):
        for value in (np.zeros((281, 1)), np.zeros((0, 20)), np.zeros(784),
                      np.array([[np.nan]]), np.array([[-1]]), np.array([[256]]), np.array([[True]])):
            with self.subTest(shape=value.shape), self.assertRaises(ValueError):
                preprocess(value)


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor(ROOT / 'checkpoints/numpy_best.npz')

    def test_prediction_and_processed_image_contract(self):
        data = payload(stroke())
        result = self.predictor.predict(data)
        probabilities = np.asarray(result['probabilities'])
        self.assertEqual(probabilities.shape, (10,))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertAlmostEqual(float(probabilities.sum()), 1, places=6)
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))
        processed = np.asarray(result['processed'], dtype=np.float32).reshape(1, 784)
        np.testing.assert_array_equal(self.predictor.model.predict_proba(processed)[0], probabilities.astype(np.float32))
        self.assertEqual(result['digit'], probabilities.argmax())
        self.assertEqual(result['confidence'], probabilities.max())
        self.assertEqual(result, self.predictor.predict(data))

    def test_blank_abstains(self):
        result = self.predictor.predict(payload(np.zeros((28, 28))))
        self.assertTrue(result['blank'])
        self.assertIsNone(result['digit'])
        self.assertIsNone(result['confidence'])
        self.assertEqual(result['probabilities'], [])

    def test_checkpoint_loaded_once(self):
        with patch('app.MLP.load', wraps=MLP.load) as load:
            p = Predictor(ROOT / 'checkpoints/numpy_best.npz')
            p.predict(payload(stroke())); p.predict(payload(stroke()))
            self.assertEqual(load.call_count, 1)

    def test_invalid_payloads(self):
        for data in (None, {}, {'width': 2, 'height': 2, 'pixels': [0]},
                     {'width': 281, 'height': 1, 'pixels': []},
                     {'width': True, 'height': 1, 'pixels': [0]},
                     {'width': 1, 'height': 1, 'pixels': [256]},
                     {'width': 1, 'height': 1, 'pixels': [1.5]},
                     {'width': 1, 'height': 1, 'pixels': [True]},
                     {'width': 1, 'height': 1, 'pixels': [0], 'path': '/tmp/model'}):
            with self.subTest(data=data), self.assertRaises(ValueError):
                self.predictor.predict(data)


class DrawingHTTPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = DrawingServer(('127.0.0.1', 0), Predictor(ROOT / 'checkpoints/numpy_best.npz'))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join()

    def request(self, method, path, body=None, headers=None):
        conn = http.client.HTTPConnection(*self.server.server_address, timeout=5)
        h = {'Content-Type': 'application/json'}
        h.update(headers or {})
        conn.request(method, path, body=body, headers=h)
        response = conn.getresponse(); status, body = response.status, response.read()
        conn.close()
        return status, body

    def test_assets_and_fixed_routes(self):
        for path in ('/', '/app.js', '/style.css'):
            status, body = self.request('GET', path)
            self.assertEqual(status, 200); self.assertGreater(len(body), 100)
        for path in ('/../checkpoints/numpy_best.npz', '/app.py', '/checkpoints/numpy_best.npz'):
            self.assertEqual(self.request('GET', path)[0], 404)

    def test_end_to_end(self):
        status, body = self.request('POST', '/api/predict', json.dumps(payload(stroke())))
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)['probabilities']), 10)
        status, body = self.request('POST', '/api/predict', json.dumps(payload(np.zeros((280, 280)))))
        self.assertEqual(status, 200); self.assertTrue(json.loads(body)['blank'])

    def test_malformed_and_oversized(self):
        for body in (b'no json', b'\xff', b'[' * 2000 + b']' * 2000, b'{"width": 1, "height": 1, "pixels": [NaN]}'):
            self.assertEqual(self.request('POST', '/api/predict', body)[0], 400)
        self.assertEqual(self.request('POST', '/api/predict', b'{}', {'Content-Length': str(MAX_BODY + 1)})[0], 413)
        self.assertEqual(self.request('POST', '/api/predict', '{}', {'Content-Type': 'text/plain'})[0], 415)
        self.assertEqual(self.request('POST', '/other', '{}')[0], 404)

    def test_origin_and_host(self):
        self.assertEqual(self.request('POST', '/api/predict', '{}', {'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.request('POST', '/api/predict', '{}', {'Host': 'attacker.test', 'Origin': 'http://attacker.test'})[0], 403)

#!/usr/bin/env python3
"""Local drawing UI and bounded, read-only model inference API."""
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(name, '1')
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from nn.model import MLP
from nn.preprocess import preprocess

ROOT = Path(__file__).resolve().parent
MAX_BODY = 400000


class Predictor:
    def __init__(self, checkpoint: Path):
        self.model = MLP.load(str(checkpoint))
        if self.model.sizes[0] != 784 or self.model.sizes[-1] != 10:
            raise ValueError('drawing interface requires 784 inputs and 10 output classes')

    def predict(self, data):
        if not isinstance(data, dict) or set(data) != {'width', 'height', 'pixels'}:
            raise ValueError('provide width, height, and a flat pixels array only')
        w, h = data['width'], data['height']
        if type(w) is not int or type(h) is not int or not 1 <= w <= 280 or not 1 <= h <= 280:
            raise ValueError('image dimensions must be integers in 1..280')
        pixels = data['pixels']
        if not isinstance(pixels, list) or len(pixels) != w * h:
            raise ValueError('pixel count does not match image dimensions')
        if any(type(p) is not int or not 0 <= p <= 255 for p in pixels):
            raise ValueError('pixels must be integer grayscale values from 0 to 255')
        x, info = preprocess(np.asarray(pixels, dtype=np.uint8).reshape(h, w))
        if info['blank']:
            return {'blank': True, 'digit': None, 'confidence': None, 'probabilities': [],
                    'processed': x.reshape(28, 28).tolist(), 'preprocessing': info}
        probabilities = self.model.predict_proba(x)[0]
        digit = int(probabilities.argmax())
        return {'blank': False, 'digit': digit, 'confidence': float(probabilities[digit]),
                'probabilities': probabilities.tolist(), 'processed': x.reshape(28, 28).tolist(),
                'preprocessing': info}


class DrawingServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, predictor):
        self.predictor = predictor
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_):
        pass

    def reply(self, status, body, content_type='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False).encode()
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def allowed(self):
        host = self.headers.get('Host')
        port = self.server.server_port
        hosts = {'localhost:%d' % port, '127.0.0.1:%d' % port}
        if port == 80:
            hosts.update({'localhost', '127.0.0.1'})
        origin = self.headers.get('Origin')
        if host not in hosts or (origin and origin != 'http://' + host):
            self.reply(403, {'error': 'Only same-origin localhost requests are allowed.'})
            return False
        return True

    def do_GET(self):
        if not self.allowed():
            return
        files = {'/': ('index.html', 'text/html; charset=utf-8'),
                 '/app.js': ('app.js', 'text/javascript; charset=utf-8'),
                 '/style.css': ('style.css', 'text/css; charset=utf-8')}
        route = urlsplit(self.path).path
        if route not in files:
            self.reply(404, {'error': 'Not found.'})
            return
        name, mime = files[route]
        self.reply(200, (ROOT / 'web' / name).read_bytes(), mime)

    def do_POST(self):
        if not self.allowed():
            return
        if self.path != '/api/predict':
            self.reply(404, {'error': 'Not found.'})
            return
        if self.headers.get('Content-Type', '').split(';')[0].strip().lower() != 'application/json':
            self.reply(415, {'error': 'Send application/json.'})
            return
        if self.headers.get('Transfer-Encoding'):
            self.reply(400, {'error': 'Chunked requests are not supported.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= MAX_BODY:
                self.reply(413, {'error': 'Request must contain 1..400000 bytes.'})
                return
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError('Incomplete request.')
            result = self.server.predictor.predict(json.loads(raw))
            self.reply(200, result)
        except (ValueError, TypeError, UnicodeError, RecursionError):
            self.reply(400, {'error': 'Invalid image request. Use integer grayscale pixels in [0,255] and dimensions up to 280×280.'})
        except TimeoutError:
            self.reply(408, {'error': 'Request timed out.'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    try:
        predictor = Predictor(ROOT / 'checkpoints' / 'numpy_best.npz')
        server = DrawingServer(('127.0.0.1', args.port), predictor)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Digit Lab: http://127.0.0.1:{args.port} (Ctrl+C to stop)', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()

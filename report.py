#!/usr/bin/env python3
"""Rebuild standalone SVG/HTML charts from measured results; no plotting packages."""
import base64
from html import escape
import json
from pathlib import Path
import struct
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'


def text(x, y, value, size=13, **attrs):
    extra = ' '.join('%s="%s"' % (k.replace('_', '-'), escape(str(v), quote=True)) for k, v in attrs.items())
    return f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{size}" {extra}>{escape(str(value))}</text>'


def svg_start(w, h, title):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img">',
            '<title>' + escape(title) + '</title>', '<rect width="100%" height="100%" fill="white"/>', text(20, 28, title, 20)]


def curves(results, metric):
    parts = svg_start(850, 420, metric.title() + ' learning curves')
    series = []
    colors = ('#1465a4', '#d27413', '#31845b', '#943b83')
    for key in ('numpy', 'pytorch'):
        if 'history' not in results[key]:
            continue
        for split in ('train', 'validation'):
            series.append((key + ' ' + split, [r[split + '_' + metric] for r in results[key]['history']]))
    ymax = 1 if metric == 'accuracy' else max(v for _, values in series for v in values) * 1.1
    epochs = max(len(values) for _, values in series)
    x = lambda i: 75 + (i - 1) * 690 / max(1, epochs - 1)
    y = lambda v: 330 - v * 260 / ymax
    for tick in range(6):
        v = tick * ymax / 5
        parts += [f'<line x1="75" y1="{y(v)}" x2="765" y2="{y(v)}" stroke="#ddd"/>',
                  text(65, y(v) + 4, f'{v:.2f}', text_anchor='end')]
    for epoch in range(1, epochs + 1):
        parts.append(text(x(epoch), 350, epoch, text_anchor='middle'))
    parts.append(text(400, 375, 'Epoch (end-of-epoch evaluation)'))
    for i, (name, values) in enumerate(series):
        points = ' '.join(f'{x(j+1):.2f},{y(v):.2f}' for j, v in enumerate(values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[i]}" stroke-width="2"/>')
        parts.append(text(60 + i * 195, 405, name, 12, fill=colors[i]))
    parts.append('</svg>')
    (OUT / (metric + '.svg')).write_text('\n'.join(parts))


def confusion(result, filename, title):
    matrix = np.asarray(result['test']['confusion_matrix'])
    parts = svg_start(620, 600, title)
    for r in range(10):
        total = matrix[r].sum()
        parts.append(text(65, 96 + 43 * r, r, text_anchor='middle'))
        parts.append(text(103 + 43 * r, 68, r, text_anchor='middle'))
        for c in range(10):
            fraction = matrix[r, c] / max(1, total)
            shade = int(255 - 180 * np.sqrt(fraction))
            parts.append(f'<rect x="{82+43*c}" y="{78+43*r}" width="42" height="42" fill="rgb({shade},{shade},255)"/>')
            parts.append(text(103 + 43 * c, 103 + 43 * r, matrix[r, c], 11, text_anchor='middle'))
    parts += [text(82, 540, 'Rows: true label. Columns: predicted label. Counts shown.'),
              text(82, 565, 'Color intensity is normalized within each true-label row.'), '</svg>']
    (OUT / filename).write_text('\n'.join(parts))


def png(image):
    pixels = np.rint(image * 255).astype(np.uint8)
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)
    raw = b''.join(b'\x00' + row.tobytes() for row in pixels)
    data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 28, 28, 8, 0, 0, 0, 0))
    data += chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b'')
    return base64.b64encode(data).decode()


def examples():
    with np.load(OUT / 'examples.npz', allow_pickle=False) as sample:
        parts = svg_start(750, 920, 'NumPy: first 36 held-out test examples')
        for i, (im, label, pred) in enumerate(zip(sample['images'], sample['labels'], sample['predictions'])):
            x, y = 15 + i % 6 * 122, 50 + i // 6 * 140
            parts.append(f'<image x="{x}" y="{y}" width="100" height="100" href="data:image/png;base64,{png(im)}"/>')
            parts.append(text(x, y + 119, f'True {label} / pred {pred}', 11, fill='#166534' if label == pred else '#b91c1c'))
        parts.append('</svg>')
    (OUT / 'predictions.svg').write_text('\n'.join(parts))


def main():
    metrics = json.loads((OUT / 'metrics.json').read_text())
    curves(metrics, 'loss'); curves(metrics, 'accuracy')
    confusion(metrics['numpy'], 'confusion-numpy.svg', 'NumPy held-out test confusion matrix')
    if 'test' in metrics['pytorch']:
        confusion(metrics['pytorch'], 'confusion-pytorch.svg', 'PyTorch held-out test confusion matrix')
    else:
        (OUT / 'confusion-pytorch.svg').unlink(missing_ok=True)
    examples()
    rows = []
    for name in ('numpy', 'pytorch'):
        m = metrics[name]
        if 'test' in m:
            rows.append('<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in
                        (name, m['selected_epoch'], f"{m['train']['accuracy']:.2%}", f"{m['validation']['accuracy']:.2%}",
                         f"{m['test']['accuracy']:.2%}", f"{m['test']['loss']:.4f}", f"{m['training_seconds']:.2f}")) + '</tr>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>MNIST experiment</title>
<style>body{font:16px system-ui;max-width:1050px;margin:30px auto;padding:12px}td,th{padding:10px;border:1px solid #ccc}table{border-collapse:collapse}img{max-width:100%;height:auto}</style>
<h1>Neural Network From Scratch — measured MNIST results</h1>
<p>784 → 128 → 64 → 10; CPU, float32; 50,000 training / 10,000 validation / 10,000 test examples. Validation loss selects the checkpoint. Test labels never enter training.</p>
<table><tr><th>Implementation</th><th>Selected epoch</th><th>Train accuracy</th><th>Validation accuracy</th><th>Test accuracy</th><th>Test loss</th><th>Training seconds</th></tr>'''
    html += ''.join(rows) + '</table><p>One seeded run; no confidence intervals or general speedup claim. See metrics.json and README for timing boundaries and limitations.</p>'
    for file in ('accuracy.svg', 'loss.svg', 'confusion-numpy.svg', 'confusion-pytorch.svg', 'predictions.svg'):
        if (OUT / file).exists():
            html += f'<p><img src="{file}" alt="{file}"></p>'
    (OUT / 'report.html').write_text(html + '</html>')
    print('Generated learning curves, confusion matrices, predictions, and results/report.html')


if __name__ == '__main__':
    main()

# Canvas → MNIST input

The original trained model expects white-on-black pixels in `[0,1]`, shape `(B,784)`, row-major order. The UI captures the canvas's intrinsic **280×280** raster (independent of its CSS display size) and sends each red channel value as a grayscale integer. White strokes on an opaque black canvas make all RGB channels equal. Browser canvas rendering antialiases stroke edges. No image-processing packages are added.

## Deterministic backend pipeline

1. Validate a finite grayscale numeric image with dimensions 1..280 and values 0..255; the HTTP contract further requires integer pixels. Normalize to float32 by dividing by 255.
2. Find the minimal bounding box enclosing pixels with intensity **strictly greater than 8/255**. If none exist, return a blank result and abstain from prediction. Fainter antialiased pixels inside the box retain their actual intensities; faint pixels outside this box are excluded.
3. Let cropped dimensions be H,W; scale by `s = 20 / max(H,W)`. Choose `H' = max(1, round(sH))` and `W' = max(1, round(sW))`, capped at 20. Aspect ratio is preserved up to integer pixel rounding.
4. Resize with separable weight matrices, `R = W_y @ crop @ W_x.T`. For shrinking, each target pixel averages exact source-pixel overlap areas, preventing aliasing. For enlargement, use bilinear interpolation at pixel centers with clamped edges. Neither nearest-neighbor downsampling nor point sampling is used.
5. Place the resized digit in a zero-filled 28×28 image. Compute intensity center of mass `c_y = Σ(y·I_yx)/ΣI`, `c_x = Σ(x·I_yx)/ΣI`. Translate by integer offsets `round(13.5−c_y)`, `round(13.5−c_x)`, clamped to retain the entire resized rectangle. No wraparound or edge clipping occurs. Integer shifts preserve intensities; centering is normally within half a pixel, but very asymmetric mass can be constrained by available margins.
6. Flatten in C/row-major order to contiguous float32 shape **(1,784)**. This is the input to the existing `MLP.predict_proba`. Return its ten softmax probabilities and argmax.

The preview displays the **same normalized values returned by the backend**, rounded only for an 8-bit canvas display. It does not silently apply a separate image transform in JavaScript.

## HTTP contract

Start `python app.py`, then `POST /api/predict` with `Content-Type: application/json` and a JSON object containing exactly:

```json
{"width": 2, "height": 2, "pixels": [0, 255, 0, 255]}
```

Pixels are row-major integers, length `width * height`. Dimensions are integers in 1..280; booleans, floats, NaN, strings, extra keys, and mismatched lengths are rejected. Maximum body: 400,000 bytes. A UI-sized image fits this bound. Nonblank responses contain `blank:false`, `digit`, `confidence`, ten `probabilities`, a 28×28 float `processed` array, and `preprocessing` metadata. Blank responses contain `blank:true`, `digit:null`, `confidence:null`, `probabilities:[]`, and an all-zero processed image. Wrong routes return 404; wrong content type 415; invalid payloads 400; invalid sizes 413; Host/Origin violations 403.

No request may specify a model or file path. The checkpoint location is fixed relative to the application. No external service receives drawings. NPZ inference parameters are loaded once before the server starts.

## Verification and practical limits

Tests cover blank/faint images, exact bounding boxes, aspect ratio, normalization/dtype/shape, center of mass, intensity conservation, translation invariance, row-major orientation, analytic antialiasing examples, deterministic predictions, probability normalization, exact preview/inference consistency, one-time loading, malformed/oversized HTTP input, and fixed-route/security behavior.

This approximates MNIST's size/centering conventions; it does not reproduce every detail of the original dataset's digit construction. There is no deskewing, denoising, stroke thinning, or handwriting adaptation. Tiny marks can be enlarged into misleading inputs. Draw one clear digit. Confidence is uncalibrated and does not measure whether an image is a digit. No canvas accuracy benchmark is claimed; existing MNIST results are unchanged.

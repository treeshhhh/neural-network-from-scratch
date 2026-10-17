# Drawing-interface verification — October 17, 2026

**READY TO USE LOCALLY.** Existing network implementation, backpropagation, checkpoints, PyTorch baseline, experiment metrics, plots, and original tests are preserved. No retraining, external API, or GitHub operation was performed.

## Actual automated results

Final `make test` using the existing project-local Python 3.12 environment: **32 test methods passed, zero failures, zero skips** (17 original + 15 new). PyTorch was available for the original optional parity test. See `ui-test-output.txt`. The first focused HTTP attempt was blocked by the environment's loopback sandbox; after permission was granted, the complete suite passed.

New checks cover blank/faint drawings; normalized `(1,784)` float32 output; bounding-box/aspect-ratio handling; center of mass without clipping or loss of intensity; antialiased area reduction and bilinear enlargement; deterministic preprocessing and translation invariance; row-major orientation; exact consistency between returned processed pixels and model probabilities; probability normalization; blank abstention; one-time checkpoint loading; malformed/dimension/type errors, deeply nested JSON, oversized HTTP bodies, content types, fixed routes, Host validation, and cross-origin rejection.

## Browser smoke check

Opened the local standard-library server in the in-app browser. Verified empty-canvas messaging. Drew a vertical digit “1” with a pointer drag: automatic prediction returned 1 and displayed ten probabilities, a 28×28 input preview, and confidence rounded to 100.0%. Clear reset the result; disabling automatic prediction left the next drawing pending until Predict was clicked. Manual Predict returned 1 with the matching preview/probability display. Screenshot: `digit-ui.png`.

This is a functional smoke check, not a recognition-accuracy study. No new canvas accuracy metric is claimed. The model can be confidently wrong, particularly for unusual handwriting, tiny marks, letters, or multiple digits. The original held-out MNIST metrics remain historical measurements of the unchanged checkpoint.

## Implementation and limits

Python standard library + existing NumPy only; vanilla HTML/CSS/JavaScript. Server loads the checkpoint once, binds to IPv4 loopback, accepts bounded grayscale JSON, and exposes no arbitrary paths or shell execution. Images are processed only in memory, with no external requests or drawing persistence. Resize/centering use one backend implementation, and the UI displays the same normalized tensor returned with inference results.

Manual browser testing covered one browser and pointer input, not every digit, device, browser engine, mobile width, failure race, or accessibility mode. Responsive CSS and mouse/trackpad/touch pointer support are implemented; only the pointer-drag path was exercised in the browser. The local development server is not intended for public hosting. Normalized input alignment improves compatibility but does not remove the MNIST-to-handwriting distribution gap.

# Verification — October 17, 2026

**READY TO PUBLISH** as a CPU-based educational/portfolio project. No GitHub repository was created or modified. Project 3 was not started.

## Automated results

Final `make test` with Python 3.12.14, NumPy 2.3.5, and optional PyTorch 2.14.1: **17 test methods passed, zero failures, zero skips**. Raw output is in `test-output.txt`.

Coverage: forward/output shapes, softmax normalization and shift invariance, finite extreme-logit loss, known cross-entropy values, all finite-difference derivatives on a small network, inactive ReLU behavior, vanilla SGD and momentum arithmetic, synthetic learning, NPZ save/load/prediction consistency, fixed-seed training, batch-independent evaluation with no parameter mutation, input rejection, IDX parsing/corruption, disjoint reproducible stratified splits, two-step PyTorch float64 gradient/update parity, included checkpoint/example consistency, recorded confusion counts, and generated SVG parsing.

Central differences checked **39 parameters** with ε=10⁻⁵ in float64; maximum absolute discrepancy was **5.65×10⁻¹²**. This independently validates the manual chain rule on a small network; it is not an exhaustive check of all 109,386 MNIST-model parameters.

Both saved checkpoints were separately reloaded and re-evaluated on all 50,000 train, 10,000 validation, and 10,000 test examples. All loss, accuracy, and confusion-matrix values exactly matched the recorded metrics on this environment. See `checkpoint-verification.json`. No retraining was performed for this check.

## Actual experiment

One seed (42), six epochs per implementation, all 50,000 training examples per epoch, architecture 784→128→64→10, batch 128, float32, SGD learning rate 0.1 and momentum 0.9. Validation has 10,000 disjoint original training examples; official test set has 10,000 examples. Selection uses lowest validation loss, never test labels.

| Model | Selected epoch | Train accuracy / loss | Validation accuracy / loss | Test accuracy / loss | Training seconds |
|---|---:|---:|---:|---:|---:|
| NumPy | 3 | 98.198% / 0.0551805 | 96.95% / 0.1044182 | 97.16% / 0.0905822 | 1.149082 |
| PyTorch-trained | 6 | 98.984% / 0.0329633 | 97.19% / 0.1007810 | 97.17% / 0.0956104 | 1.058994 |

These are observed `perf_counter` wall times on this execution environment, not expected runtimes for a user's Mac. They include epoch train/validation evaluations and in-memory checkpoint selection; exclude downloads, setup/imports, final test evaluation, disk checkpoint writing, and plotting. NumPy ran first, PyTorch second. No repetitions, timing warmup study, hyperparameter search, or confidence intervals. No general speedup conclusion is supported. NumPy used Accelerate BLAS; thread limits were requested but actual BLAS thread concurrency was not independently measured. PyTorch CPU threads were set to one.

Training configurations, complete epoch histories, environment metadata, and full confusion counts are in `metrics.json`; console output is preserved in `training-log.txt`. Both models start from the same weights and use identical splits/permutations. Their float32 numerical kernels differ. Baseline final metrics are computed with the common NumPy evaluator on exported PyTorch-trained parameters; per-epoch baseline metrics use native PyTorch.

## Visual check and packaging

Opened `report.html` using a local HTTP server and inspected the measured-results table, learning curves, and confusion matrices in the browser. SVG files parse correctly and use local data; example predictions are the first 36 official test images, not selected for correctness. Static report contains no external requests or paid dependencies. No exhaustive browser compatibility testing was performed.

The archive includes complete source, math/design/learning/interview/resume documentation, 17 tests, recorded results and charts, split indices, 36 example images, and two small NPZ inference checkpoints. Full MNIST files, the project-local installation environment, caches, and credentials are excluded.

Remaining limits: one seed and modest training budget; no statistical uncertainty, robustness tests, arbitrary-photo preprocessing, exact optimizer-state resume, or cross-platform execution verification. Model weights are meaningful trained artifacts, not a claim of independent authorship or research novelty.

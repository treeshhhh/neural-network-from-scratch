# Architecture

`nn/model.py` contains the entire NumPy model, manually derived gradients, SGD/momentum, and NPZ parameter persistence. It never imports PyTorch. `nn/data.py` downloads and integrity-checks the original MNIST gzip/IDX files, normalizes pixels, and generates explicit split/permutation indices. `nn/evaluate.py` computes sample-weighted loss, accuracy, predictions, and confusion counts without updating parameters.

`train.py` orchestrates one configured experiment, records per-epoch metrics, chooses the lowest-validation-loss checkpoint, and only then evaluates held-out test data. No early stopping, augmentation, dropout, regularization, or hyperparameter search is used. The supplied run trains all 50,000 training examples for six epochs. Validation contains 1,000 examples per digit sampled from the original 60,000 training examples; the official 10,000 test examples are separate files. Split indices are included for auditability. Test data never influences updates or checkpoint selection.

`baseline.py` separately builds an equivalent CPU PyTorch Sequential network, copying the same initial NumPy weights (transposing dense matrices to PyTorch's convention). Both models receive identical normalized float32 arrays, split indices, per-epoch permutations, batch size, epoch count, learning rate, and classical momentum. The baseline uses autograd only in this separate module. Checkpoints use a common NPZ weight layout, with pickled object loading disabled. Baseline final metrics use the common NumPy evaluation routine on exported PyTorch-trained parameters; its per-epoch metrics use native PyTorch. Tiny rounding differences can change activation masks and produce diverging optimization paths over thousands of batches. Exact final parameter equality is not expected.

`predict.py` loads a supplied checkpoint to evaluate MNIST or classify normalized `.npy` rows. `report.py` generates static SVG charts and a local HTML summary from stored metrics, with no plotting dependency or web server. `tests/` contains finite differences, optimizer/serialization checks, synthetic-data learning, IDX/split checks, and an optional PyTorch reference test.

## Artifacts and reproducibility

`results/metrics.json` is the machine-readable experiment record; `training-log.txt` is actual console output; `split_indices.npz` records the original training-set row indices; `examples.npz` stores the first 36 normalized test images, labels, and predictions. Checkpoints contain model dimensions and weight/bias arrays only; optimizer buffers are intentionally omitted, so exact training resume is unsupported. Loading validates dimensions, shapes, numeric dtype, finite parameters, and expected keys.

A fixed seed controls initialization and batching. Reproducibility is strongest with the recorded package versions and hardware; bitwise equality across BLAS libraries/CPU architectures is not promised. PyTorch is configured for one thread and deterministic operations; NumPy thread-control environment variables request one BLAS thread, but actual internal BLAS concurrency was not independently measured.

## Timing boundaries

`time.perf_counter()` measures elapsed training-loop wall time, including minibatches, end-of-epoch full training/validation evaluation, and in-memory copies of the best parameters. Dataset downloads/loading, import/model/optimizer setup, final held-out evaluation, disk checkpoint writes, and plotting are excluded. Frameworks run sequentially (NumPy first), once each, with no warmup trials. Consequently timings describe this run, not a statistically controlled speed comparison. NumPy used Apple's Accelerate BLAS; PyTorch uses its own kernels. Both are CPU-only; no GPU or paid service is involved.

## Local drawing application

`app.py` constructs one `Predictor` at startup and reuses its existing checkpoint for every request. `web/` is a vanilla canvas UI; it sends bounded 8-bit grayscale samples to `/api/predict`. `nn/preprocess.py` alone owns crop/resize/centering, giving tests and the browser a single source of truth. The endpoint returns the exact float32 normalized input in its response alongside ten probabilities and the argmax. A blank image returns an explicit abstention, not an arbitrary class.

Browser pointer capture handles strokes outside the canvas boundary. A request-version counter and AbortController prevent old responses from repopulating results after Clear or new drawing. Optional idle prediction is debounced; overlapping requests are canceled in the browser. The standard-library HTTP server binds to IPv4 loopback, applies Host/Origin checks, caps request bytes/dimensions, and uses fixed routes. Inference is read-only; model parameters and all original experiment artifacts are untouched. There is no server-side file upload, shell execution, external inference service, or drawing persistence. See PREPROCESSING.md for the endpoint contract and limitations.

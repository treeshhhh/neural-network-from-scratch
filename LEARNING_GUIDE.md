# Learning guide

1. Read `MATH.md` alongside `MLP._forward`. For a batch of 128, write down every matrix shape: 128×784, 784×128, 128×128, 128×64, 64×10. Explain bias broadcasting.
2. Implement a two-class softmax on paper. Show why adding a constant to every logit leaves probabilities unchanged. Explain why log-sum-exp is safer than `-log(probability)`.
3. Derive `(P−Y)/B` from the softmax Jacobian and cross-entropy. Then derive `A.T @ delta` by differentiating one weight. Trace how gradients flow backward through three dense layers.
4. Run `make test`. Study the finite-difference loop: it perturbs one parameter at a time using float64. Why does ε too small cause cancellation? Why must checks avoid ReLU's kink at zero?
5. Read the optimizer. With μ=0.9 and constant gradient 1, calculate the first three velocity values. Explain why gradients must be computed before changing the weights.
6. Open `results/report.html`. Compare the epoch with highest validation accuracy to the epoch with lowest validation loss. They need not match: loss measures confidence as well as correctness. The supplied NumPy checkpoint is selected at epoch 3, not the final epoch.
7. Inspect the confusion matrix. A row is the true digit; columns are predictions. Compute recall for one digit and explain why overall accuracy hides which pairs are confused.
8. Load the checkpoint and run predictions. Explain the distinction between inference parameters and optimizer state. Add a test before modifying either format.

## Practical experiments after understanding the baseline

Change only one factor, such as hidden width or learning rate, and record validation results. Keep test labels untouched while choosing configurations. Do not repeatedly optimize against the test set; that leaks evaluation information even without explicit gradient updates. Multiple seeds would quantify uncertainty, but are intentionally omitted from this resource-bounded run.

Try fitting a small synthetic dataset before training MNIST. If loss increases, check gradient sign, averaging, label order, and the learning rate before adding complexity. If loss becomes NaN, inspect logits and matrix scales. Random chance for ten balanced classes is roughly 10%; a model performing well on training but poorly on validation may be overfitting or facing a distribution mismatch.

This implementation is an AI-assisted starting point. To make it your portfolio work, explain every equation, run the tests, and implement a small change you can defend. Do not claim research novelty or production use.

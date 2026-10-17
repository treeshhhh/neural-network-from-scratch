# Interview preparation

1. **What is implemented from scratch?** Dense forward propagation, ReLU, stable softmax/logit cross-entropy, manual backpropagation, initialization, momentum, batching, checkpointing, evaluation. NumPy supplies arrays and matrix multiplication, not gradients.
2. **Why softmax on logits only once?** Cross-entropy accepts logits and computes log-sum-exp. Applying softmax first changes the expected input and can lose numerical stability.
3. **Derive the output gradient.** `∂loss/∂logits = (probabilities − one_hot_labels)/batch_size`. Show the softmax Jacobian cancellation.
4. **Why `A.T @ delta`?** Each weight's derivative is a sum of input activation × downstream gradient over examples. State its dimensions.
5. **What happens at ReLU zero?** The derivative is undefined; choose zero. Numerical gradient tests should avoid kinks.
6. **Why He initialization?** Variance 2/fan_in compensates approximately for ReLU suppressing half the inputs. Biases can start equal because weights break symmetry.
7. **Why mini-batches?** They balance vectorized throughput and stochastic gradient estimates. The last partial batch must divide by its own size.
8. **How did you check gradients?** All 39 scalar parameters of a small deep network were compared with central finite differences; an optional independent autograd comparison verifies two momentum steps.
9. **How is leakage prevented?** The original training partition is split into disjoint train/validation indices; optimizer updates only use train. Lowest validation loss selects a checkpoint before accessing official test metrics.
10. **Why NumPy and PyTorch results differ despite matched setup?** Different float32 kernels/reduction orders cause rounding differences that accumulate and change ReLU masks. No claim of identical training trajectories is made.
11. **What does training time include?** Training batches, epoch train/validation evaluation, and best-parameter copies; excludes downloads, setup, final test evaluation, checkpoint disk I/O, and plots. One sequential run cannot establish a general speed ranking.
12. **Can training resume exactly?** No. Saved NPZ stores inference parameters and dimensions, not momentum/RNG progress. Extending this requires explicit state/versioning tests.
13. **What is a limitation of the evaluation?** One seed and six epochs, no confidence intervals, no robustness or external handwriting dataset. MNIST performance does not imply general handwriting recognition.

## Demo in two minutes

Show the NumPy gradient code, one finite-difference test, a learning curve, and a confusion matrix. Explain why the best-validation checkpoint can precede the last epoch. Predict using the included weights without retraining. State measured metrics and the comparison limitations accurately.

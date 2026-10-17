# Resume bullets

Use after studying and adapting the implementation; describe AI assistance accurately. No research novelty, deployment, or general speedup is claimed.

- Implemented a 109,386-parameter NumPy neural network with manual backpropagation, stable softmax/cross-entropy, mini-batch SGD with momentum, and non-pickle checkpoint persistence.
- Trained a 784→128→64→10 classifier on 50,000 MNIST examples with a separate validation partition, achieving 97.16% held-out test accuracy in one six-epoch seeded experiment.
- Validated derivatives with finite differences and an independent PyTorch autograd reference, and compared against a matched PyTorch baseline achieving 97.17% test accuracy.

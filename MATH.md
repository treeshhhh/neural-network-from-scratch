# Mathematics and numerical checks

Use row-major batches: B examples, d₀ inputs, K classes, L dense layers. Parameters Wₗ ∈ ℝ^(dₗ₋₁ × dₗ), bₗ ∈ ℝ^(dₗ). For MNIST, dimensions are 784, 128, 64, 10; there are 109,386 trainable scalar parameters.

## Forward pass

A₀ = X ∈ ℝ^(B × 784).

Zₗ = Aₗ₋₁ Wₗ + 1 bₗᵀ.

Aₗ = ReLU(Zₗ) for hidden layers; final Z_L are logits. ReLU(z) = max(0,z), applied elementwise.

For row i, pᵢⱼ = exp(zᵢⱼ − mᵢ) / Σₖ exp(zᵢₖ − mᵢ), with mᵢ = maxⱼ zᵢⱼ. Subtracting a constant does not change softmax. Each probability row sums to 1.

For integer labels yᵢ, mean cross-entropy is:

ℒ = (1/B) Σᵢ [log Σⱼ exp(zᵢⱼ − mᵢ) − (zᵢ,yᵢ − mᵢ)].

Compute loss from logits via log-sum-exp, rather than taking log of a possibly underflowed probability. No arbitrary epsilon clipping changes the derivative. For logits [1000, −1000] and target 1, finite loss is 2000 even though the target probability underflows to zero.

## Derivative of softmax with cross-entropy

For one example, ∂pⱼ/∂zₖ = pⱼ(δⱼₖ − pₖ). Since ℓ = −log p_y,

∂ℓ/∂zₖ = −(1/p_y) p_y(δ_yk − pₖ) = pₖ − 1[k=y].

Thus D_L = (P − Y_onehot)/B ∈ ℝ^(B × K). The implementation subtracts one at the target index without allocating a full one-hot array.

## Backpropagation through a dense layer

From dZ = dA W + A dW + 1 dbᵀ and the trace/Frobenius inner product:

∇Wₗ ℒ = Aₗ₋₁ᵀ Dₗ ∈ ℝ^(dₗ₋₁ × dₗ).

∇bₗ ℒ = Σᵢ Dₗ[i,:] ∈ ℝ^(dₗ).

Dₗ₋₁ = (Dₗ Wₗᵀ) ⊙ 1[Zₗ₋₁ > 0] for a preceding hidden layer.

ReLU's derivative is 1 for z>0 and 0 for z<0. It is undefined at zero; this implementation chooses 0, a valid subgradient. Gradients use the forward-pass weights; all optimizer updates happen after backpropagation completes. Divide by B exactly once, in D_L; additional averaging would shrink gradients incorrectly.

## Initialization and optimization

Hidden weights: independent normal entries with variance 2/fan_in (He initialization) to counteract ReLU's removal of roughly half the signal. Output linear weights use variance 1/fan_in. Biases start at zero. Separate random weights break hidden-unit symmetry.

SGD with classical momentum, initially V=0:

Vₜ = μVₜ₋₁ + ∇θℒₜ; θₜ₊₁ = θₜ − ηVₜ.

μ=0 recovers vanilla SGD. Included experiment uses η=0.1, μ=0.9, batch size 128; the last, shorter batch uses its actual B. Momentum state is not stored in inference checkpoints.

## Complexity

Per batch, dense forward/backward costs O(B Σₗ dₗ₋₁dₗ); backpropagation adds a constant factor to forward matrix multiplication. Parameters occupy O(Σₗ dₗ₋₁dₗ); cached activations/preactivations require O(B Σₗ dₗ). Momentum adds one parameter-sized buffer. Epoch cost is O(N Σₗ dₗ₋₁dₗ). The experiment uses float32; gradient checking uses float64 to reduce finite-difference cancellation.

## Independent validation

Central difference g_numeric = [ℒ(θ+εeⱼ)−ℒ(θ−εeⱼ)]/(2ε), ε=10⁻⁵. The test checks all 39 parameters of a 3→4→3→2 network, avoiding ReLU kinks. Observed maximum absolute discrepancy: 5.65×10⁻¹². Tests also compare logits, loss, gradients, and two momentum updates with PyTorch float64 autograd. These are numerical checks, not a formal proof for every possible input.

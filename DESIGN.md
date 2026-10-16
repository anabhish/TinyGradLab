# TinyGradLab Design

## 1. Overview

TinyGradLab is a small neural-network and automatic-differentiation library implemented using Python and NumPy. It includes a custom reverse-mode automatic-differentiation engine, a multilayer perceptron, classification losses, pruning methods, and an SGD optimizer with momentum.

The core implementation does not rely on PyTorch, TensorFlow, JAX, or Keras. The project is intended to demonstrate the mechanics of automatic differentiation, model training, weight pruning, and sparse computation.

## 2. Automatic Differentiation

The `Tensor` class stores numerical data, gradients, parent tensors, and the operation that created each result. During backpropagation, the engine traverses the computation graph in topological order and propagates gradients using the chain rule.

Gradients from multiple paths are accumulated because a tensor may contribute to several downstream operations. Broadcasted operations reduce gradients back to the original operand shapes. Matrix multiplication uses the corresponding matrix-gradient rules.

The implementation includes elementwise arithmetic, matrix multiplication, reductions, and nonlinear activations. Finite-difference tests compare analytical gradients with numerical approximations for supported operations.

### Autodiff bottleneck

The main bottleneck is the Python-level computation graph and backward execution. Each operation creates tensor objects and backward closures, and graph traversal and gradient accumulation introduce Python overhead. This is convenient for understanding reverse-mode automatic differentiation, but it is less efficient than a system that fuses operations or executes kernels in optimized native code.

NumPy performs the underlying array operations efficiently, but the surrounding graph management and backward logic remain Python-driven. The current implementation prioritizes clarity and correctness over execution speed.

## 3. Neural Network and Training

The neural network consists of two linear layers with a ReLU activation between them. The linear layers initialize weights using a He-style scale:

\[
\text{scale} = \sqrt{\frac{2}{\text{fan-in}}}
\]

This initialization is intended to help preserve activation variance when using ReLU. Biases are initialized to zero.

Binary cross-entropy with logits uses a numerically stable formulation. The implementation also includes numerically stable softmax cross-entropy for classification.

The training example uses mini-batches, shuffles examples each epoch, and updates parameters using momentum SGD. The XOR experiment is a small educational example rather than a benchmark of generalization.

## 4. Pruning

TinyGradLab supports global magnitude pruning, Taylor-saliency pruning, and progressive pruning. Binary masks identify weights that have been pruned. Masked parameters remain zero during optimizer updates.

### 4.1 Magnitude criterion

Magnitude pruning ranks weights by absolute value:

\[
s_i = |w_i|
\]

Weights with the smallest scores are selected for removal until the target sparsity is reached.

### 4.2 Taylor-saliency criterion

Taylor-saliency pruning estimates the local effect of removing a weight using a first-order Taylor approximation of the loss.

For a parameter \(w_i\), setting it to zero changes it by \(\Delta w_i = -w_i\). The first-order approximation is:

\[
\Delta L_i \approx
\frac{\partial L}{\partial w_i}\Delta w_i
= -w_i\frac{\partial L}{\partial w_i}
\]

The implementation ranks parameters using the absolute score:

\[
s_i =
\left|w_i\frac{\partial L}{\partial w_i}\right|
\]

Weights with lower scores are considered less locally important and are selected for pruning first. This is an approximation based on the current loss and gradients; it does not guarantee that pruning those weights will improve the final model or outperform magnitude pruning.

### 4.3 Progressive pruning

Progressive pruning increases the target sparsity over multiple scheduled steps instead of removing all target weights at once. Previously pruned weights remain masked, and additional active weights are selected at subsequent steps.

The multi-seed experiment compares one-shot magnitude, progressive magnitude, one-shot Taylor saliency, and progressive Taylor saliency at multiple target sparsities.

### 4.4 Masked-weight gradient semantics
A masked connection is constrained to contribute zero to the forward computation. In the current implementation, the optimizer enforces this constraint during `SGD.step()` by masking the parameter values, gradients, and momentum. Consequently, the gradient stored immediately after `backward()` may be nonzero for a masked connection; it is zeroed when the optimizer applies the mask. When a connection is revived, its momentum is reset before the update so stale momentum cannot corrupt its training. This defines the masking contract at the optimizer-step boundary.
## 5. Experimental Findings

Experiments on the four-example XOR training dataset compare four pruning methods across five random seeds and target sparsities of 0%, 50%, 75%, 90%, and 95%.

The latest recorded mean training accuracies were:

| Target sparsity | One-shot magnitude | Progressive magnitude | One-shot Taylor | Progressive Taylor |
|---|---:|---:|---:|---:|
| 0% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 75% | 100% | 100% | 90% | 95% |
| 90% | 75% | 95% | 55% | 70% |
| 95% | 55% | 70% | 50% | 45% |

These figures are mean training accuracy across five seeds, not validation or test accuracy. The dataset is extremely small, so these results do not establish generalization to unseen examples.

In this experiment, progressive magnitude pruning performed better than progressive Taylor-saliency pruning at 90% and 95% sparsity. Taylor saliency did not consistently outperform magnitude pruning. These results are specific to the chosen model, training procedure, dataset, and seeds.

The experiment produces `pruning_trials.csv`, `pruning_results.csv`, and `pruning_pareto.png` to preserve individual trial results, aggregated results, and the accuracy-versus-sparsity plot.

## 6. Computational Cost and Sparse Execution

The benchmark compares NumPy dense matrix multiplication with the project's Python-loop sparse implementation for a batch size of 32, 128 input features, and 128 output features. The benchmark checks numerical agreement between both paths and reports median per-call timing over five rounds of 30 repetitions.

In the latest recorded run, dense runtimes were approximately 0.113 ms, 0.125 ms, and 0.115 ms at 0%, 50%, and 75% sparsity, respectively. Sparse runtimes were approximately 67.195 ms, 33.163 ms, and 16.044 ms. The corresponding speedup ratios were 0.00169×, 0.00377×, and 0.00715×.

These are measurements from one machine and one run, not universal performance guarantees. The sparse path is slower because Python-loop overhead outweighs the computation saved by skipping zero values. This project does not claim an inference speedup.
## 7. Serving a Self-Pruned Model in a Multi-Tenant Inference Service

A self-pruned model could be served as a versioned model artifact containing its parameter values, masks, architecture configuration, and relevant metadata. The serving layer would load the artifact and use an execution path that respects the stored masks.

In a multi-tenant service, model artifacts and inference state should be isolated by tenant and model version. Requests should be routed to the correct model instance, and one tenant's model updates or pruning state must not modify another tenant's parameters. Resource limits, request batching, access control, and model-version tracking would also be needed.

The current TinyGradLab implementation is a research and educational prototype; it does not implement a production multi-tenant serving system. Its sparse multiplication prototype is not yet suitable evidence of production inference performance.

## 8. Limitations

- Experiments use a four-example XOR dataset.
- Reported accuracies are training accuracies.
- Taylor saliency is a first-order local approximation and does not guarantee better pruning results.
- The sparse implementation is educational rather than performance-optimized.
- Benchmark timings depend on the machine, workload, and NumPy backend.
- The current results do not demonstrate a general sparse-inference speedup.
- Multi-tenant serving is a design discussion, not an implemented service.
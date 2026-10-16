# TinyGradLab: Design and Experimental Evaluation

## 1. Overview

TinyGradLab is a small neural-network and automatic-differentiation library implemented using Python and NumPy. It includes a custom reverse-mode automatic-differentiation engine, a multilayer perceptron, classification losses, pruning methods, and an SGD optimizer with momentum.

The core implementation does not rely on PyTorch, TensorFlow, JAX, or Keras. The project is intended to demonstrate the mechanics of automatic differentiation, model training, weight pruning, and sparse computation.

This document describes the implementation, the pruning evaluation protocol, measured outcomes, computational-cost observations, and limitations. The experiments are deliberately small and reproducible; their results characterize this setup rather than establish general performance or generalization claims.

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

## 4. Pruning Methods and Experimental Evaluation

### 4.1 Importance Criterion: First-Order Taylor Saliency

Magnitude pruning ranks weights by absolute value, \(s_i=|w_i|\), and is retained as a baseline. The primary method uses first-order Taylor saliency.

Let \(L\) be the loss, \(w_i\) a connection weight, and \(g_i=\partial L/\partial w_i\). Removing a connection changes its weight to zero, so \(\Delta w_i=-w_i\). A first-order expansion gives:

\[
\Delta L_i \approx \frac{\partial L}{\partial w_i}\Delta w_i
= -w_i g_i
\]

The ranking score is the absolute estimated loss change:

\[
s_i = |w_i g_i|
\]

Active connections are ranked globally and the lowest scores are pruned first. This is a local approximation based on the current batch's loss and gradients; it ignores higher-order terms and interactions among connections. It therefore motivates a candidate ranking but does not guarantee better final accuracy than magnitude pruning.

### 4.2 Progressive Cubic Schedule

The training demonstration uses a 40-hidden-unit MLP on the four-example XOR dataset. Its two weight matrices contain 120 connections. The pruning target ramps from zero at epoch 500 to 90% at epoch 1500, with updates every 50 epochs. The target follows:

\[
s(t)=s_f\left[1-(1-p)^3\right],
\quad
p=\frac{t-t_0}{t_f-t_0},
\quad
p\in[0,1]
\]

where \(s_f=0.90\). This cubic form increases sparsity more rapidly early in the interval and reduces the size of increments near the target. The model continues training after epoch 1500 with the final mask fixed.

The multi-seed comparison uses the same cubic target ramp for progressive magnitude and progressive Taylor methods. One-shot methods prune at epoch 500 and continue training for the remaining budget; they are included to compare progressive adaptation against the simpler but more abrupt alternative.

### 4.3 Mask Semantics and Optimizer Invariants

Masks are binary arrays for the weight matrices. Pruning sets the mask to zero and multiplies the corresponding parameter by the mask, so removed weights are exact zeros in subsequent forward passes. The SGD optimizer masks gradients, momentum, and parameters at update time, preventing optimizer updates from reviving pruned weights. Progressive pruning only considers active mask entries, so previously pruned connections remain pruned.

The tests cover Taylor score ordering, progressive mask preservation, exact zero values, and optimizer behavior. Regrowth is not implemented: permanent masks make the experiment easier to reason about and avoid the instability of reactivating weights with stale optimizer state.

### 4.4 Exact Sparsity Budget

Sparsity is computed over the two weight matrices only; biases are excluded. The multi-seed experiment checks that the mask has exactly the requested number of zeros, using integer target counts. The 120-weight experiment can represent 0%, 50%, 75%, 90%, and 95% exactly. The main training example asserts that it reaches 90% and checks that every masked weight is exactly zero.

### 4.5 Multi-Seed Comparison and Accuracy–Sparsity Trade-off

`examples/multi_seed_experiment.py` compares four methods at target sparsities of 0%, 50%, 75%, 90%, and 95% using seeds 11, 22, 33, 44, and 55:

- One-shot magnitude.
- Progressive magnitude.
- One-shot Taylor.
- Progressive Taylor.

All methods use the same 40-hidden-unit model, XOR examples, optimizer settings, 2,000-epoch budget, and seeds. Final loss is evaluated on the full four-example dataset. The script writes:

- `pruning_trials.csv`: all 100 seed/method/target trials.
- `pruning_results.csv`: means and sample standard deviations across five seeds.
- `pruning_paired_comparison.csv`: per-seed Taylor-minus-magnitude accuracy differences and approximate normal 95% intervals.
- `pruning_pareto.png`: mean accuracy against achieved sparsity, with ±1 sample standard deviation.

These are training-set measurements on four XOR examples, not held-out test results. Five seeds give a small sample; uncertainty intervals are descriptive and should not be interpreted as strong statistical proof.

#### Measured mean training accuracy

| Target sparsity | One-shot magnitude | Progressive magnitude | One-shot Taylor | Progressive Taylor |
|---:|---:|---:|---:|---:|
| 0% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 75% | 100% | 100% | 90% | 100% |
| 90% | 75% | 100% | 55% | 100% |
| 95% | 55% | 90% | 50% | 70% |

At 90% sparsity, progressive Taylor and progressive magnitude both achieved 100% training accuracy in all five seeds. Their paired mean accuracy difference is 0 percentage points (sample SD 0 percentage points). At 95%, progressive Taylor averaged 70% versus 90% for progressive magnitude, a paired mean difference of -20 percentage points (approximate normal 95% interval: -29.8 to -10.2 percentage points). This small-seed interval is descriptive, and the results do not show Taylor consistently outperforming magnitude.

**Falsifiable claim:** With the recorded setup and seeds 11, 22, 33, 44, and 55, progressive Taylor pruning reaches 90% weight sparsity and 100% XOR training accuracy in all five trials, matching progressive magnitude at that target. The claim is falsified if rerunning the documented experiment under the same environment and settings produces a different result. It is not a claim about generalization.

### 4.6 Computational Cost: Pruned Model and Sparse-Aware Execution

`src/tinygradlab/engine/sparse.py` implements `sparse_matmul`, which iterates over nonzero matrix entries instead of calling dense matrix multiplication on a matrix containing zeros. `examples/benchmark_pruning.py` compares dense NumPy multiplication with this sparse-aware path, checks numerical equivalence, and records active weights, estimated multiply-accumulate counts (MACs), and median wall-clock timing.

The benchmark records a 128 × 128 matrix-kernel sweep at 0%, 50%, 75%, 90%, and 95% sparsity in `pruning_cost_benchmark.csv`. It also loads the actual model exported by `examples/train_mlp.py` and records a full-XOR-batch measurement in `pruned_xor_model_cost.csv`. Run the training example before the benchmark so `pruned_xor_model.npz` exists.

In the recorded run, the trained XOR model had 12 active weights out of 120 (90% sparsity). For a batch of four examples, the weight-based MAC estimate fell from 480 dense MACs to 48 sparse-aware MACs. The median dense forward time was 0.0049 ms, while the Python-loop sparse-aware path took 0.0285 ms, a dense/sparse time ratio of about 0.173×. Thus the sparse path reduced the estimated multiply-accumulate work but was about 5.8 times slower in wall-clock time for this tiny model.

For the 128 × 128 kernel at 90% target sparsity, the recorded run had 1,639 active weights and an estimated 52,448 sparse-aware MACs per batch versus 524,288 dense MACs. Dense NumPy took about 0.021 ms; the Python-loop sparse path took about 2.946 ms. These measurements are machine- and implementation-specific. They demonstrate that fewer active operations do not automatically yield lower latency in Python. **No wall-clock speedup is claimed.**

### 4.7 Reproducibility

From the project root, run:

```powershell
python -m pytest -q
python examples/train_mlp.py
python examples/multi_seed_experiment.py
python examples/benchmark_pruning.py
python -m pytest -q
git diff --check
```

The scripts regenerate the training curve, pruning trial and summary CSVs, paired-comparison CSV, Pareto plot, and cost CSVs. Timings will vary by machine.

## 5. Deployment Considerations for Multi-Tenant Inference

A self-pruned model could be served as a versioned model artifact containing its parameter values, masks, architecture configuration, and relevant metadata. The serving layer would load the artifact and use an execution path that respects the stored masks.

In a multi-tenant service, model artifacts and inference state should be isolated by tenant and model version. Requests should be routed to the correct model instance, and one tenant's model updates or pruning state must not modify another tenant's parameters. Resource limits, request batching, access control, and model-version tracking would also be needed.

The current TinyGradLab implementation is a research and educational prototype; it does not implement a production multi-tenant serving system. Its sparse multiplication prototype is not yet suitable for production inference performance.

## 6. Limitations

- Experiments use a four-example XOR dataset and report training accuracy, not held-out generalization.
- Taylor saliency is a first-order local approximation and does not consistently outperform magnitude pruning.
- Five seeds provide limited evidence; standard deviations and paired intervals are descriptive.
- The sparse-aware implementation is written as a Python loop and is slower than optimized dense NumPy in the measured workloads.
- MAC counts describe estimated multiply-accumulate work, not end-to-end runtime.
- Multi-tenant serving is a design discussion, not an implemented service.

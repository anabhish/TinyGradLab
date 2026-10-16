# TinyGradLab

TinyGradLab is a lightweight neural-network and automatic-differentiation library built from scratch using Python and NumPy. It demonstrates reverse-mode automatic differentiation, neural-network training, momentum SGD, weight pruning, and experimental sparse computation.

## Features

- Custom `Tensor` class and computation graph for reverse-mode automatic differentiation.
- Gradient propagation and accumulation.
- Elementwise arithmetic, broadcasting, matrix multiplication, sum, and mean operations.
- ReLU, tanh, sigmoid, and GELU activations.
- Numerically stable binary cross-entropy with logits and softmax cross-entropy.
- Multilayer perceptron (MLP) for binary classification.
- Mini-batch training with shuffled examples.
- Momentum stochastic gradient descent (SGD).
- Magnitude-based and Taylor-saliency-based weight pruning.
- Progressive pruning with binary masks.
- Mask-aware optimizer updates.
- Sparse matrix multiplication prototype.
- Finite-difference gradient checks and unit tests.
- Multi-seed pruning experiments, CSV results, and a Pareto plot.

## Requirements

- Python 3.10 or compatible.
- NumPy.
- pytest for tests.
- Matplotlib for experiment plots.

The core implementation uses Python and NumPy. Scikit-learn, if used, is permitted only for dataset loading; it is not used for model training or automatic differentiation.

## Installation

From the project root, create and activate a virtual environment if needed.

For PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the project and its dependencies:

```powershell
python -m pip install -e .
python -m pip install pytest matplotlib
```

## Run tests

Run the complete test suite:

```powershell
python -m pytest -q
```

Run the tensor and autodiff tests, including finite-difference checks:

```powershell
python -m pytest tests/test_tensor.py -q
```

Run the loss tests, including softmax cross-entropy gradient checks:

```powershell
python -m pytest tests/test_losses.py -q
```

The latest recorded full test run passed **67 tests**.

## Part 2 — Train the MLP

Run the training example:

```powershell
python examples/train_mlp.py
```

The example trains an MLP on the four-example XOR dataset using mini-batches of two examples. It shuffles the examples each epoch, uses momentum SGD, and progressively prunes weights at target sparsities of 25%, 50%, and 75%.

The script reports training loss, accuracy, and weight sparsity during training.

The linear layers use He-style weight initialization with scale \(\sqrt{2 / \text{fan-in}}\), which is suitable for helping preserve activation variance with ReLU. Biases are initialized to zero.

## Part 3 — Compare pruning methods

Run the pruning comparison:

```powershell
python examples/compare_pruning.py
```

This example compares pruning behavior using the project's pruning utilities.

The implementation includes:
- **Magnitude pruning:** ranks weights by absolute value.
- **Taylor-saliency pruning:** ranks weights using the first-order score \(|w \cdot \partial L/\partial w|\).
- **Progressive pruning:** increases sparsity over multiple scheduled steps while retaining previously pruned weights.

Taylor saliency estimates a weight's local contribution to loss using a first-order Taylor approximation. It is an approximation, not a guarantee that the resulting model will have lower loss or higher accuracy.

## Part 4 — Multi-seed Pareto sweep

Run the multi-seed experiment:

```powershell
python examples/multi_seed_experiment.py
```

The experiment evaluates four methods across five target sparsities and five random seeds:

- One-shot magnitude pruning.
- Progressive magnitude pruning.
- One-shot Taylor-saliency pruning.
- Progressive Taylor-saliency pruning.

The target sparsities are 0%, 50%, 75%, 90%, and 95%.

The script generates three evidence files in the current working directory:

- `pruning_trials.csv` — individual trial results.
- `pruning_results.csv` — aggregated results.
- `pruning_pareto.png` — Pareto plot of accuracy against sparsity.

These files are intended to support reproducible comparison of pruning methods.

### Experimental results

The latest recorded five-seed experiment on the four-example XOR training dataset produced the following mean training accuracies:

| Target sparsity | One-shot magnitude | Progressive magnitude | One-shot Taylor | Progressive Taylor |
|---|---:|---:|---:|---:|
| 0% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 75% | 100% | 100% | 90% | 95% |
| 90% | 75% | 95% | 55% | 70% |
| 95% | 55% | 70% | 50% | 45% |

These are mean training accuracies across five seeds, not validation or test accuracies. The XOR dataset contains only four examples, so these results do not establish generalization to unseen data.

In this experiment, progressive magnitude pruning outperformed progressive Taylor-saliency pruning at 90% and 95% sparsity. Taylor saliency did not consistently outperform magnitude pruning. The results are specific to this model, dataset, training setup, and random seeds.

## Computational-cost benchmark

Run the dense-versus-sparse matrix multiplication benchmark:

```powershell
python examples/benchmark_pruning.py
```

The project includes a sparse matrix multiplication prototype for exploring computation with sparse data.

**Important limitation:** setting weights to zero does not automatically accelerate dense matrix multiplication. In the current benchmark, the Python-loop sparse implementation was substantially slower than NumPy's optimized dense operation, even at high sparsity.

The sparse implementation is a correctness-oriented prototype, not a performance-optimized inference backend. No inference speedup is claimed. Sparsity and computational speed are separate measurements, and the benchmark should be rerun on the target machine when reporting timing results.

## Additional experiments

Compare plain SGD and momentum SGD:

```powershell
python examples/compare_optimizers.py
```

The optimizer experiment compares training behavior on the project's small dataset. Its results should be interpreted as educational observations rather than evidence of generalization.

## Project structure

```text
TinyGradLab/
├── examples/
│   ├── train_mlp.py
│   ├── compare_pruning.py
│   ├── multi_seed_experiment.py
│   ├── compare_optimizers.py
│   └── benchmark_pruning.py
├── src/
│   └── tinygradlab/
│       ├── engine/
│       ├── nn/
│       └── optim/
├── tests/
├── DESIGN.md
├── README.md
└── pyproject.toml
```

## Scope and limitations

TinyGradLab is an educational implementation intended to demonstrate automatic differentiation, neural-network training, pruning, and sparse computation. The core engine and model training are implemented using Python and NumPy rather than established deep-learning or automatic-differentiation frameworks.

The experiments use a very small XOR dataset. Their accuracy results are not evidence of performance on larger datasets. The sparse matrix multiplication implementation is a prototype, and actual speedups require a sparse-aware execution path and separate benchmarking.

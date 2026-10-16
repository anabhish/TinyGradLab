# TinyGradLab

A lightweight neural-network and automatic-differentiation library built from scratch using Python and NumPy.

TinyGradLab implements a reverse-mode automatic-differentiation engine, a multilayer perceptron, binary classification loss, pruning algorithms, and SGD with momentum.

## Features

- Custom `Tensor` class and computation graph
- Reverse-mode automatic differentiation and gradient propagation
- Broadcasting-aware arithmetic and matrix multiplication
- ReLU, tanh, sigmoid, and GELU activations
- Numerically stable binary cross-entropy with logits
- MLP for binary classification
- Global magnitude pruning and progressive pruning
- Binary pruning masks and masked optimizer state
- Sparse matrix multiplication prototype
- Unit tests and reproducible experiment scripts

## Requirements

- Python 3.10 or a compatible Python version
- NumPy
- pytest for tests
- Matplotlib for experiment plots

## Setup

Activate your virtual environment, then install the project and its dependencies according to `pyproject.toml`.

For example, from the project root in PowerShell:

```powershell
python -m pip install -e .
python -m pip install pytest matplotlib
```

## Run tests

```powershell
python -m pytest -q
```

The current test suite contains 49 passing tests.

## Run experiments

Train the MLP:

```powershell
python examples/train_mlp.py
```

Compare one-shot and progressive pruning:

```powershell
python examples/compare_pruning.py
```

Run the multi-seed pruning experiment:

```powershell
python examples/multi_seed_experiment.py
```

Compare plain SGD and momentum SGD:

```powershell
python examples/compare_optimizers.py
```

Benchmark dense versus sparse matrix multiplication:

```powershell
python examples/benchmark_pruning.py
```

The multi-seed experiment writes `pruning_trials.csv`, `pruning_results.csv`, and `pruning_pareto.png` to the current working directory.

## Experimental observations

On the four-example XOR training dataset, the five-seed experiment reported:

| Target sparsity | One-shot mean accuracy | Progressive mean accuracy |
|---|---:|---:|
| 25% | 100% | 100% |
| 50% | 100% | 100% |
| 75% | 85% | 95% |

In a separate optimizer experiment, both plain SGD and momentum SGD achieved 100% mean training accuracy. Momentum SGD reported a lower mean loss in that run.

These are small educational experiments on the training dataset, not evidence of generalization to unseen data.

## Sparse computation limitation

Setting weights to zero does not automatically accelerate dense matrix multiplication. In the current benchmark, the Python-loop sparse implementation was much slower than NumPy's optimized dense operation, even at 75% sparsity.

The sparse implementation is currently a correctness-oriented prototype, not a performance-optimized inference backend.

## Project structure

```text
TinyGradLab/
├── examples/
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

## Scope

This project is intended for learning and experimentation. It does not depend on PyTorch, TensorFlow, JAX, or Keras for its implementation.

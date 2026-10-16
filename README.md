# TinyGradLab

TinyGradLab is a small neural-network and automatic-differentiation library implemented with Python and NumPy. It demonstrates reverse-mode autodiff, MLP training, momentum SGD, pruning, and experimental sparse computation.

## Features

- Custom reverse-mode `Tensor` engine with finite-difference gradient tests.
- Numerically stable binary cross-entropy with logits and softmax cross-entropy.
- MLP training with mini-batches and momentum SGD.
- Magnitude and Taylor-saliency pruning, including progressive pruning and binary masks.
- Sparse-aware matrix multiplication prototype.
- Multi-seed pruning comparisons, raw CSV evidence, Pareto plot, and cost benchmarks.

## Requirements and installation

Python 3.10+, NumPy, pytest, and Matplotlib are required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest matplotlib
```

## Tests

```powershell
python -m pytest -q
```

The revised test suite passes **77 tests** in the project environment used for this revision.

## progressive pruning and evidence

### Importance criterion

The primary method uses first-order Taylor saliency. For weight \(w_i\) and loss gradient \(g_i=\partial L/\partial w_i\), removing the connection changes the weight by \(-w_i\). A first-order Taylor approximation gives \(\Delta L_i\approx-w_i g_i\), so the importance score is \(S_i=|w_i g_i|\). The lowest-scoring active weights are pruned first.

This estimates local loss sensitivity; it is not a guarantee that Taylor pruning will outperform magnitude pruning. Magnitude pruning is retained as a baseline.

### Progressive schedule and masks

The main training example uses a 40-hidden-unit MLP on four XOR examples. Its 120 weight connections are progressively pruned from a dense warm-up to 90% target sparsity using a cubic target ramp from epochs 500 to 1500, with updates every 50 epochs. Training continues after the target is reached.

Masks set pruned weights to exact zero. The optimizer masks gradients, momentum, and parameters so removed weights cannot be accidentally restored. The main training script checks the final 90% sparsity target and exact zeros, and exports `pruned_xor_model.npz` for cost measurement.

### Reproduce and the Pareto experiment

```powershell
python examples/train_mlp.py
python examples/multi_seed_experiment.py
```

The multi-seed experiment compares one-shot magnitude, progressive magnitude, one-shot Taylor, and progressive Taylor at 0%, 50%, 75%, 90%, and 95% target sparsity using seeds 11, 22, 33, 44, and 55.

Committed evidence files:

- `pruning_trials.csv` — all individual seed/method/target results.
- `pruning_results.csv` — mean accuracy and loss, sample standard deviations, achieved sparsity, and active-weight counts.
- `pruning_paired_comparison.csv` — paired Taylor-minus-magnitude accuracy differences and approximate normal 95% intervals.
- `pruning_pareto.png` — mean accuracy versus achieved sparsity with ±1 sample standard deviation.
- `training_curve.png` — loss, accuracy, and sparsity history for the main training example.

#### Recorded mean training accuracy

| Target sparsity | One-shot magnitude | Progressive magnitude | One-shot Taylor | Progressive Taylor |
|---:|---:|---:|---:|---:|
| 0% | 100% | 100% | 100% | 100% |
| 50% | 100% | 100% | 100% | 100% |
| 75% | 100% | 100% | 90% | 100% |
| 90% | 75% | 100% | 55% | 100% |
| 95% | 55% | 90% | 50% | 70% |

**Falsifiable claim:** In the recorded experiment, progressive Taylor pruning reached 90% weight sparsity and 100% XOR training accuracy in all five seeds, matching progressive magnitude pruning at that target. The paired mean accuracy difference at 90% was 0 percentage points (sample SD 0). At 95%, progressive Taylor averaged 70% versus 90% for progressive magnitude. These are results on four training examples, not evidence of generalization; the five-seed sample is small, and Taylor did not consistently outperform magnitude pruning.

### Real computation-cost measurement

```powershell
python examples/benchmark_pruning.py
```

Run `train_mlp.py` first; the benchmark then measures both a 128 × 128 sparse-kernel sweep and the actual exported pruned XOR MLP. The sparse-aware path in `src/tinygradlab/engine/sparse.py` iterates over nonzero weights rather than doing dense matrix multiplication over zeros. Outputs are checked against the dense path.

The benchmark writes:

- `pruning_cost_benchmark.csv` — active weights, estimated MACs, and dense/sparse-aware timing for the 128 × 128 kernel.
- `pruned_xor_model_cost.csv` — active weights, estimated MACs, and timing for the trained pruned XOR model.

In the recorded run, the XOR model had 12 active weights out of 120 (90% sparsity). Estimated MACs for four examples fell from 480 dense MACs to 48 sparse-aware MACs. Median wall-clock time was about 0.0049 ms for dense NumPy versus 0.0285 ms for the Python-loop sparse path. Therefore, this implementation **reduced estimated arithmetic work but did not improve latency**. No speedup is claimed. Benchmark timing is machine-specific.

## Other examples

```powershell
python examples/compare_pruning.py
python examples/compare_optimizers.py
```

## Project structure

```text
TinyGradLab/
├── examples/
│   ├── train_mlp.py
│   ├── multi_seed_experiment.py
│   ├── benchmark_pruning.py
│   ├── compare_pruning.py
│   └── compare_optimizers.py
├── src/tinygradlab/
│   ├── engine/
│   ├── nn/
│   └── optim/
├── tests/
├── pruning_trials.csv
├── pruning_results.csv
├── pruning_paired_comparison.csv
├── pruning_pareto.png
├── pruning_cost_benchmark.csv
├── pruned_xor_model_cost.csv
├── DESIGN.md
└── README.md
```

## Limitations

- The dataset contains only four XOR examples; reported accuracy is training accuracy.
- Taylor saliency is a local first-order approximation and is not consistently better than magnitude pruning.
- Five seeds provide limited statistical evidence.
- The Python-loop sparse path is slower than optimized dense NumPy in the measured workloads.
- The project is educational and does not implement a production multi-tenant serving system.

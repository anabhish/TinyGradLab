# TinyGradLab

**An engineering research prototype for reverse-mode automatic
differentiation, training-time neural-network pruning, and sparse
computation with Python and NumPy.**

TinyGradLab is a small neural-network library designed to make core
training mechanisms inspectable. It combines a custom reverse-mode
autodiff engine, an MLP training workflow, mask-based pruning methods,
and experiments that compare accuracy, sparsity, estimated arithmetic
work, and measured runtime.

The emphasis is on mathematical transparency, implementation
correctness, and reproducible measurements---not production throughput.

## Research focus

This project explores five engineering questions:

1.  How can a minimal reverse-mode autodiff engine propagate gradients
    correctly through a computation graph?
2.  How do magnitude-based and first-order Taylor-saliency criteria
    compare when selecting weights for removal?
3.  Does progressive pruning preserve training accuracy better than
    one-shot pruning?
4.  How do active parameter counts and estimated multiply-accumulate
    operations relate to measured runtime?
5.  What conclusions are supported by experiments across random seeds,
    and where should those conclusions remain limited?

## Implementation overview

  -----------------------------------------------------------------------
  Area                                Implementation
  ----------------------------------- -----------------------------------
  Automatic differentiation           Custom reverse-mode `Tensor`
                                      engine, gradient accumulation, and
                                      finite-difference tests

  Loss functions                      Numerically stable binary
                                      cross-entropy with logits and
                                      softmax cross-entropy

  Model training                      Mini-batched MLP training with
                                      momentum SGD

  Weight pruning                      Magnitude and Taylor-saliency
                                      criteria, one-shot and progressive
                                      schedules

  Mask handling                       Binary masks enforce exact zeros;
                                      optimizer updates respect masks

  Sparse computation                  Sparse-aware matrix multiplication
                                      prototype

  Evaluation                          Multi-seed comparisons, raw CSV
                                      results, Pareto plot, and cost
                                      benchmarks
  -----------------------------------------------------------------------

The core implementation uses Python and NumPy. It does not use PyTorch,
TensorFlow, JAX, Keras, or an external automatic-differentiation,
pruning, or neural-network library. Scikit-learn, if used, is limited to
loading a standard dataset; it is not used for model training, gradient
computation, or optimization.

## Environment and installation

Python 3.10+, NumPy, pytest, and Matplotlib are required.

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install pytest matplotlib
```

The editable install command assumes the repository contains a valid
packaging configuration such as `pyproject.toml` or `setup.py`. If that
configuration is not present, add it before running `pip install -e .`.

## Validation and experiment workflow

Run commands from the repository root.

``` powershell
python -m pytest -q
python examples/train_mlp.py
python examples/multi_seed_experiment.py
python examples/benchmark_pruning.py
```

The training script exports `pruned_xor_model.npz`; run it before the
cost benchmark so the benchmark can measure the actual trained model.

The revised test suite passed **77 tests** in the project environment
used for this revision. Re-run the suite in the environment used for any
new measurements.

## Training-time pruning design

Pruning occurs during training under a configured schedule. At each
pruning update, an importance criterion ranks active weights, masks are
updated, and training continues with the remaining effective
connections.

### Magnitude importance

Magnitude pruning ranks each weight (w_i) by its absolute value:

\[ s_i = \|w_i\|. \]

Weights with the smallest scores are removed first. This is a simple
baseline, but magnitude alone does not directly estimate the loss change
caused by removing a connection.

### First-order Taylor saliency

Let (L) denote the loss, (w_i) a connection weight, and (g_i =
`\partial `{=tex}L/`\partial `{=tex}w_i) its gradient. Removing a
connection changes the weight by (`\Delta `{=tex}w_i=-w_i). A
first-order Taylor approximation gives

\[ `\Delta `{=tex}L_i `\approx `{=tex}g_i`\Delta `{=tex}w_i=-w_i g_i. \]

The importance score is the absolute estimated loss change:

\[ S_i = \|w_i g_i\|. \]

The score estimates local loss sensitivity around the current parameter
values. It ignores higher-order terms and interactions among
connections, so it is a ranking heuristic rather than a guarantee that
Taylor pruning will outperform magnitude pruning.

### Progressive schedule and masks

The main training example uses a 40-hidden-unit MLP on four XOR
examples. Its two weight matrices contain 120 connections. The target
sparsity increases from a dense warm-up to 90% using a cubic ramp from
epoch 500 to epoch 1500, with updates every 50 epochs. Training
continues after the target is reached.

The cubic schedule is

\[ s(t)=s_f`\left[1-(1-p)^3\right]`{=tex}, `\qquad`{=tex}
p=`\frac{t-t_0}{t_f-t_0}`{=tex},`\quad `{=tex}p`\in[0,1]`{=tex}, \]

where (s_f=0.90), (t_0=500), and (t_f=1500).

Masks set pruned weights to exact zero. The optimizer masks gradients,
momentum, and parameters so that optimizer updates do not accidentally
restore removed weights. The main training script checks the final 90%
sparsity target and exact zeros, then exports `pruned_xor_model.npz` for
cost measurement. Regrowth is not implemented.

## Experimental protocol

The multi-seed experiment compares four methods at target sparsities of
0%, 50%, 75%, 90%, and 95%, using seeds 11, 22, 33, 44, and 55:

-   One-shot magnitude pruning
-   Progressive magnitude pruning
-   One-shot Taylor pruning
-   Progressive Taylor pruning

The experiment writes raw per-trial results, aggregate statistics,
paired comparisons, and a Pareto plot. Accuracy is measured on the four
XOR training examples, not on held-out data.

### Recorded mean training accuracy

  --------------------------------------------------------------------------
          Target       One-shot    Progressive       One-shot    Progressive
        sparsity      magnitude      magnitude         Taylor         Taylor
  -------------- -------------- -------------- -------------- --------------
              0%           100%           100%           100%           100%

             50%           100%           100%           100%           100%

             75%           100%           100%            90%           100%

             90%            75%           100%            55%           100%

             95%            55%            90%            50%            70%
  --------------------------------------------------------------------------

In the recorded experiment, progressive Taylor pruning reached 90%
weight sparsity and 100% XOR training accuracy in all five seeds,
matching progressive magnitude pruning at that target. The paired mean
accuracy difference at 90% was zero percentage points (sample standard
deviation zero). At 95%, progressive Taylor averaged 70% versus 90% for
progressive magnitude.

These measurements support a narrow claim about the recorded setup. They
do not show that Taylor saliency consistently outperforms magnitude
pruning. The dataset contains only four examples, and five seeds provide
limited statistical evidence.

**Falsifiable claim:** In the recorded experiment, progressive Taylor pruning reached 90% weight sparsity and 100% XOR training accuracy in all five seeds, matching progressive magnitude pruning at that target. The paired mean accuracy difference at 90% was 0 percentage points (sample SD 0). At 95%, progressive Taylor averaged 70% versus 90% for progressive magnitude. These are results on four training examples, not evidence of generalization; the five-seed sample is small, and Taylor did not consistently outperform magnitude pruning.

## Cost model and runtime measurements

The sparse-aware path in `src/tinygradlab/engine/sparse.py` iterates
over nonzero weights rather than performing dense matrix multiplication
on a matrix containing zeros. The benchmark checks its outputs against
the dense path and records active weights, estimated multiply-accumulate
operations (MACs), and timing.

Run:

``` powershell
python examples/benchmark_pruning.py
```

The benchmark measures both a 128 × 128 sparse-kernel sweep and the
actual exported pruned XOR MLP. It writes:

-   `pruning_cost_benchmark.csv` --- active weights, estimated MACs, and
    dense/sparse-aware timings for the 128 × 128 kernel.
-   `pruned_xor_model_cost.csv` --- active weights, estimated MACs, and
    timings for the trained XOR model.

In the recorded run, the XOR model had 12 active weights out of 120 (90%
sparsity). For four examples, estimated MACs fell from 480 dense MACs to
48 sparse-aware MACs. Median wall-clock time was approximately 0.0049 ms
for dense NumPy and 0.0285 ms for the Python-loop sparse path.

Therefore, the implementation reduced estimated arithmetic work but did
not improve latency in this measurement. The Python-loop overhead
outweighs the saved arithmetic for this small workload. No wall-clock
speedup is claimed; timings are machine-specific.

## Generated artifacts

  -----------------------------------------------------------------------
  Artifact                            Description
  ----------------------------------- -----------------------------------
  `training_curve.png`                Training loss, accuracy, and
                                      sparsity history

  `pruning_trials.csv`                Individual seed/method/target
                                      results

  `pruning_results.csv`               Mean accuracy and loss, sample
                                      standard deviations, achieved
                                      sparsity, and active-weight counts

  `pruning_paired_comparison.csv`     Paired Taylor-minus-magnitude
                                      accuracy differences and
                                      approximate normal 95% intervals

  `pruning_pareto.png`                Mean accuracy versus achieved
                                      sparsity, with ±1 sample standard
                                      deviation

  `pruning_cost_benchmark.csv`        Dense versus sparse-aware cost
                                      measurements for the 128 × 128
                                      kernel

  `pruned_xor_model_cost.csv`         Cost measurements for the exported
                                      trained XOR model

  `pruned_xor_model.npz`              Exported model parameters and masks
  -----------------------------------------------------------------------

## Additional experiments

``` powershell
python examples/compare_pruning.py
python examples/compare_optimizers.py
```

## Repository structure

``` text
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

-   The dataset contains only four XOR examples; reported accuracy is
    training accuracy.
-   Taylor saliency is a local first-order approximation and is not
    consistently better than magnitude pruning.
-   Five seeds provide limited statistical evidence.
-   The Python-loop sparse path is slower than optimized dense NumPy in
    the measured workloads.
-   MAC counts estimate arithmetic work and do not directly predict
    end-to-end latency.
-   Pruned connections do not regrow.
-   TinyGradLab is an educational research prototype and does not
    implement a production multi-tenant serving system.

## Future work

Useful follow-up investigations include broader datasets and held-out
evaluation, additional random seeds, alternative sensitivity criteria,
profiling and vectorizing sparse kernels, and experiments with
optimizer-state policies. Any claimed performance improvement should be
measured against a clearly defined baseline and reported separately from
theoretical operation counts.

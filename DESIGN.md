# TinyGradLab Design

## 1. Overview

TinyGradLab is a small neural-network and automatic-differentiation library implemented using Python and NumPy. It includes a custom reverse-mode automatic-differentiation engine, a multilayer perceptron, binary classification loss, pruning methods, and an SGD optimizer with momentum.

## 2. Automatic Differentiation

The `Tensor` class stores numerical data, gradients, parent tensors, and the operation that created each result. During backpropagation, the engine traverses the computation graph in topological order and propagates gradients using the chain rule.

Broadcasted operations reduce gradients back to the original operand shapes.

## 3. Neural Network

The neural network consists of two linear layers with a ReLU activation between them. Binary cross-entropy with logits uses a numerically stable formulation for the loss.

## 4. Pruning

TinyGradLab supports global magnitude pruning and progressive pruning. Binary masks keep pruned parameters at zero during optimizer updates. Momentum state is also masked, and newly revived connections have their previous momentum reset.

## 5. Experimental Findings

Experiments on the XOR dataset compare one-shot and progressive pruning across five random seeds.

At 75% target sparsity, the observed mean training accuracy was 85% for one-shot pruning and 95% for progressive pruning. These results are specific to this small training dataset and do not establish generalization performance.

A separate benchmark compares dense NumPy matrix multiplication with a simple Python-loop sparse implementation. At 75% sparsity, the measured dense runtime was 0.145 ms and the sparse runtime was 16.684 ms for the tested matrix dimensions.

The sparse implementation is slower because Python-loop overhead outweighs the computation saved by skipping zero weights.

## 6. Limitations

- Experiments use a very small XOR dataset.
- Accuracy is measured on the training data.
- The sparse implementation is educational rather than performance-optimized.
- Benchmark timings depend on the machine and NumPy backend.
- The current results do not demonstrate a general sparse-inference speedup.

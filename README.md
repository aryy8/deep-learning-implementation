# Neural Network from Scratch

A pure NumPy implementation of a feedforward neural network, demonstrating core deep learning concepts from first principles.

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundations](#mathematical-foundations)
  - [Single Neuron](#single-neuron)
  - [Dense Layer](#dense-layer)
  - [Activation Functions](#activation-functions)
  - [Loss Functions](#loss-functions)
  - [Backpropagation](#backpropagation)
- [Network Architecture](#network-architecture)
- [Evaluation Metrics](#evaluation-metrics)

---

## Overview

This notebook builds a 2-layer neural network step-by-step:
1. Single neuron → Dense layer → Stacked layers
2. Forward propagation → Backward propagation → Training loop
3. MSE regression → Softmax classification

---

## Mathematical Foundations

### Single Neuron

A single neuron computes a weighted sum of inputs plus a bias:

$$
\begin{align*}
z &= \sum_{i=1}^{n} w_i x_i + b = \mathbf{w}^\top \mathbf{x} + b
\end{align*}
$$

Where:
- $\mathbf{x} \in \mathbb{R}^n$ — input vector
- $\mathbf{w} \in \mathbb{R}^n$ — weight vector
- $b \in \mathbb{R}$ — bias term

### Dense Layer

For a batch of $m$ samples with $n$ inputs and $k$ neurons:

$$
\begin{align*}
\mathbf{Z} &= \mathbf{X}\mathbf{W} + \mathbf{b}
\end{align*}
$$

Where:
- $\mathbf{X} \in \mathbb{R}^{m \times n}$ — input matrix
- $\mathbf{W} \in \mathbb{R}^{n \times k}$ — weight matrix
- $\mathbf{b} \in \mathbb{R}^{1 \times k}$ — bias vector (broadcast)
- $\mathbf{Z} \in \mathbb{R}^{m \times k}$ — output (pre-activation)

### Activation Functions

**ReLU (Rectified Linear Unit):**

$$
\begin{align*}
\text{ReLU}(z) &= \max(0, z)
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \text{ReLU}}{\partial z} &= \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
\end{align*}
$$

**Softmax (for classification):**

$$
\begin{align*}
\text{softmax}(z_i) &= \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}
\end{align*}
$$

### Loss Functions

**Mean Squared Error (MSE):**

$$
\begin{align*}
\mathcal{L}_{\text{MSE}} &= \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \hat{y}} &= \frac{2}{m}(\hat{y} - y)
\end{align*}
$$

**Cross-Entropy Loss:**

$$
\begin{align*}
\mathcal{L}_{\text{CE}} &= -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_{ik} \log(\hat{y}_{ik})
\end{align*}
$$

**Softmax + Cross-Entropy Gradient (combined):**

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} &= \frac{1}{m}(\hat{\mathbf{Y}} - \mathbf{Y})
\end{align*}
$$

### Backpropagation

For a dense layer with input $\mathbf{X}$ and upstream gradient $\frac{\partial \mathcal{L}}{\partial \mathbf{Z}}$:

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} &= \frac{1}{m} \mathbf{X}^\top \frac{\partial \mathcal{L}}{\partial \mathbf{Z}}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{b}} &= \frac{1}{m} \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial \mathbf{Z}_i}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{X}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{Z}} \mathbf{W}^\top
\end{align*}
$$

**ReLU Backward:**

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \mathbf{Z}} &= \frac{\partial \mathcal{L}}{\partial \mathbf{A}} \odot \mathbf{1}_{[\mathbf{Z} > 0]}
\end{align*}
$$

---

## Network Architecture

```
Input (2) → Dense (32) → ReLU → Dense (3) → Softmax → Output
```

**Training:**
- Optimizer: SGD with learning rate $\eta = 0.1$
- Mini-batch size: 64
- Epochs: 200

**Weight Update:**

$$
\begin{align*}
\mathbf{W} &\leftarrow \mathbf{W} - \eta \frac{\partial \mathcal{L}}{\partial \mathbf{W}}
\end{align*}
$$

---

## Evaluation Metrics

| Metric | Formula |
|--------|---------|
| **Accuracy** | $\frac{1}{m}\sum_{i=1}^{m} \mathbf{1}[\hat{y}_i = y_i]$ |
| **Precision** | $\frac{TP}{TP + FP}$ |
| **Recall** | $\frac{TP}{TP + FN}$ |
| **F1 Score** | $\frac{2 \cdot P \cdot R}{P + R}$ |
| **ECE** | $\sum_{b=1}^{B} \frac{n_b}{m} \|\text{acc}(b) - \text{conf}(b)\|$ |

**Final Results:**
- Accuracy: 97.3%
- Macro F1: 0.973
- Top-2 Accuracy: 100%

---

## Requirements

```
numpy
```
# deep-learning-implementation
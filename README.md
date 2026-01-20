# Deep Learning Implementations

Comprehensive notes and runnable implementation for three neural network builds:
- A NumPy-first network built from primitives.
- A Keras MLP for customer churn prediction.
- A Keras CNN for CIFAR-10 image classification.

## Index
1. [NumPy Feedforward Network (`neuralnetwork.ipynb`)](#1-numpy-feedforward-network-neuralnetworkipynb)
2. [Keras ANN for Churn (`artificialnn.ipynb`)](#2-keras-ann-for-churn-artificialnnipynb)
3. [Keras CNN for CIFAR-10 (`cnn.ipynb`)](#3-keras-cnn-for-cifar-10-cnnipynb)
4. [Repository Layout](#repository-layout)
5. [Setup and Running](#setup-and-running)

---

## 1) NumPy Feedforward Network (`neuralnetwork.ipynb`)
Purpose: implement a neural network from first principles (no deep learning framework).

Key components:
- Layers: manual Dense, ReLU, Softmax with cached tensors for backprop.
- Losses/metrics: MSE for regression; cross-entropy + softmax for classification; accuracy, confusion matrix, per-class precision/recall/F1, macro/weighted F1, top-2 accuracy, log loss, confidence, and expected calibration error (ECE).
- Training: minibatch SGD.

Architectures:

**1. Regression Model Architecture:**
```
Input(2) -> Dense(16) -> ReLU -> Dense(1)
```

**2. Classification Model Architecture:**
```
Input(2) -> Dense(32) -> ReLU -> Dense(3) -> Softmax
```
_These diagrams illustrate the sequential layers and their connections within the respective models._

Workflow (classification example):
1) Generate 3-class 2D blobs (`make_blobs`).
2) One-hot encode labels; split into train/val.
3) Forward pass: Dense -> ReLU -> Dense -> Softmax.
4) Compute cross-entropy loss.
5) Backprop manually (dZ2, dW/db, dZ1, etc.).
6) Update weights with SGD.
7) Validate and report metrics every few epochs (val accuracy ~97%).

What is demonstrated:
- From-scratch forward/backward math.
- Minibatch training loop structure.
- Rich evaluation including calibration (ECE) and top-k.

---

## 2) Keras ANN for Churn (`artificialnn.ipynb`)
Purpose: predict churn (binary classification) on `Churn_Modelling.csv` with an MLP.

Data prep:
- Select features columns 3–12; label column 13.
- One-hot encode `Geography` and `Gender` (drop_first=True).
- Scale features with `StandardScaler`.
- Train/test split (80/20).

Model Architecture:
```
Input (~10 features after encoding)
  -> Dense(8, ReLU)
  -> Dense(6, ReLU)
  -> Dense(4, ReLU)
  -> Dense(1, Sigmoid)
```
**Loss Function:** `binary_crossentropy`
**Optimizer:** `Adam`
**Callbacks:** `EarlyStopping` on `val_loss`

_This diagram shows the sequential flow of data through the layers of the Artificial Neural Network._

Training/evaluation:
- Fit with validation split and early stopping.
- Predict on test set, threshold at 0.5.
- Report confusion matrix and accuracy.

---

## 3) Keras CNN for CIFAR-10 (`cnn.ipynb`)
Purpose: classify CIFAR-10 images with a small ConvNet.

Data prep:
- Load CIFAR-10 via `tf.keras.datasets.cifar10`.
- Normalize pixel values to [0, 1].
- Visualize sample images.

Model Architecture:
```
Input (32x32x3)
  -> Conv2D(32, 3x3, ReLU)
  -> MaxPool(2x2)
  -> Conv2D(64, 3x3, ReLU)
  -> MaxPool(2x2)
  -> Conv2D(64, 3x3, ReLU)
  -> Flatten
  -> Dense(64, ReLU)
  -> Dense(10, logits)
```
**Loss Function:** `SparseCategoricalCrossentropy(from_logits=True)`
**Optimizer:** `Adam`

_This diagram outlines the layers of the Convolutional Neural Network used for image classification._

Training/evaluation:
- Train for 10 epochs with validation on the test set.
- Plot train vs. validation accuracy.
- Test accuracy around 0.69 in the provided run.

---

## Repository Layout
- `neuralnetwork.ipynb` — NumPy-first principles network with full training loop and metrics.
- `artificialnn.ipynb` — Churn prediction MLP with preprocessing and early stopping.
- `cnn.ipynb` — CIFAR-10 ConvNet example.
- `Churn_Modelling.csv` — dataset for the churn notebook.

---

## Setup and Running
1) Install dependencies (Python 3.10+ recommended):
```
pip install numpy pandas scikit-learn matplotlib tensorflow
```
2) Launch notebooks (Jupyter example):
```
jupyter notebook
```
3) Open a notebook and run top-to-bottom:
   - `neuralnetwork.ipynb`: runs entirely in NumPy; no external data required.
   - `artificialnn.ipynb`: ensure `Churn_Modelling.csv` is in the repo root.
   - `cnn.ipynb`: CIFAR-10 downloads automatically on first run.

Notes:
- GPU is optional but speeds up the Keras notebooks.
- Each notebook is self-contained; no extra Python modules beyond the listed deps.

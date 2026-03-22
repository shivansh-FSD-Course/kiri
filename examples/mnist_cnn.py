"""
examples/mnist_cnn.py

Classic MNIST digit classification — the assignment everyone gets.
Run this on a Mac Air M1/M2 with 8GB RAM. No GPU needed.

  python examples/mnist_cnn.py
"""

import numpy as np
import kiri
import kiri.nn as nn


# ── 1. load data ─────────────────────────────────────────────────────────────

def load_mnist():
    """Downloads MNIST via sklearn or falls back to a tiny synthetic dataset."""
    try:
        from sklearn.datasets import fetch_openml
        print("Loading MNIST...")
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int32)
        X = X.reshape(-1, 1, 28, 28)
        X_train, X_test = X[:60000], X[60000:]
        y_train, y_test = y[:60000], y[60000:]
        return X_train, y_train, X_test, y_test
    except Exception:
        print("sklearn not available — using synthetic data for demo")
        rng = np.random.default_rng(42)
        X_train = rng.random((1000, 1, 28, 28), dtype=np.float32)
        y_train = rng.integers(0, 10, 1000, dtype=np.int32)
        X_test  = rng.random((200, 1, 28, 28), dtype=np.float32)
        y_test  = rng.integers(0, 10, 200,  dtype=np.int32)
        return X_train, y_train, X_test, y_test


# ── 2. define model ───────────────────────────────────────────────────────────

class MnistCNN(kiri.Model):
    def __init__(self):
        self.conv1   = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1   = nn.ReLU()
        self.conv2   = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2   = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1     = nn.Linear(32 * 7 * 7, 128)
        self.relu3   = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2     = nn.Linear(128, 10)

    def forward(self, x):
        # (N, 1, 28, 28) -> (N, 16, 28, 28)
        x = self.relu1(self.conv1(x))
        # max pool 2x2 manually via strided conv trick or slice
        x = x[:, :, ::2, ::2]                 # simple 2x2 downsample
        # (N, 16, 14, 14) -> (N, 32, 14, 14)
        x = self.relu2(self.conv2(x))
        x = x[:, :, ::2, ::2]                 # (N, 32, 7, 7)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


# ── 3. train ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_mnist()

    model = MnistCNN()

    print(f"\nTraining on {len(X_train)} samples...\n")

    history = model.fit(
        X_train, y_train,
        epochs=5,
        lr=1e-3,
        batch_size=64,
        val_data=(X_test, y_test),
        verbose=True,
    )

    acc = model.accuracy(X_test, y_test)
    print(f"\n✓ Test accuracy: {acc*100:.2f}%")

    # save / load demo
    model.save("mnist_cnn.npz")
    model.load("mnist_cnn.npz")
    print("Save/load works ✓")

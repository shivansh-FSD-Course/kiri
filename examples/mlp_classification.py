"""
examples/mlp_classification.py

Simple MLP for tabular classification (iris, titanic, etc.)
The kind of assignment that should just work on any laptop.

  python examples/mlp_classification.py
"""

import numpy as np
import kiri
import kiri.nn as nn


class MLP(kiri.Model):
    def __init__(self, input_size, hidden_size, num_classes):
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # Iris dataset
    try:
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler

        iris = load_iris()
        X, y = iris.data.astype(np.float32), iris.target.astype(np.int32)
        X = StandardScaler().fit_transform(X).astype(np.float32)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    except ImportError:
        rng = np.random.default_rng(42)
        X_train = rng.random((120, 4), dtype=np.float32)
        y_train = rng.integers(0, 3, 120, dtype=np.int32)
        X_test  = rng.random((30, 4), dtype=np.float32)
        y_test  = rng.integers(0, 3, 30, dtype=np.int32)

    model = MLP(input_size=4, hidden_size=64, num_classes=3)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        lr=1e-3,
        batch_size=16,
        val_data=(X_test, y_test),
    )

    acc = model.accuracy(X_test, y_test)
    print(f"\n✓ Test accuracy: {acc*100:.1f}%")

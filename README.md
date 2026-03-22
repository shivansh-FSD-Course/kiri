# 🌫️ Kiri

**Lightweight ML for everyone. No CUDA required.**

[![PyPI version](https://img.shields.io/pypi/v/kiri-ml)](https://pypi.org/project/kiri-ml/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Kiri is a Python deep learning framework that runs natively on **Apple Silicon (M1/M2/M3/M4)** and falls back gracefully to CPU on any machine. Built for students and developers who want to train real models without a $3000 gaming PC.

---

## The problem

You're in an ML course. The assignment asks you to train a CNN on MNIST. Your classmates with gaming rigs are done in 5 minutes. You have a MacBook Air or a budget laptop. You either wait 3 hours, crash out of memory, or give up.

**Kiri fixes this.**

---

## Install

```bash
# Apple Silicon (M1/M2/M3/M4) — Metal GPU acceleration
pip install kiri-ml[apple]

# Everything else (Intel Mac, Windows, Linux) — CPU
pip install kiri-ml
```

---

## Quick start

```python
import kiri
import kiri.nn as nn
import numpy as np
```

On import, Kiri auto-detects your hardware:

```
╭─ Kiri 🌫️ ─────────────────────────────╮
│  Backend  : Apple Silicon (MLX)        │
│  Chip     : arm64                      │
│  Memory   : 16GB unified memory        │
│  Status   : ✓ Metal GPU + CPU active   │
╰────────────────────────────────────────╯
  Kiri v0.1.0  │  backend: mlx
```

### Train a model

```python
class MLP(kiri.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
history = model.fit(X_train, y_train, epochs=10, lr=1e-3, batch_size=64)

acc = model.accuracy(X_test, y_test)
print(f"Test accuracy: {acc*100:.1f}%")
```

### With DataLoader and scheduler

```python
loader    = kiri.DataLoader((X_train, y_train), batch_size=64, shuffle=True)
optimizer = kiri.optim.Adam(model, lr=1e-3)          # pass model on Apple Silicon
scheduler = kiri.optim.CosineAnnealingLR(optimizer, T_max=10)

history = model.fit(
    loader,
    epochs=10,
    optimizer=optimizer,
    scheduler=scheduler,
    val_data=(X_test, y_test),
)
```

---

## What's included

### Layers
| Layer | Description |
|-------|-------------|
| `nn.Linear(in, out)` | Fully connected |
| `nn.Conv2d(in, out, k)` | 2D convolution |
| `nn.MaxPool2d(k)` | Max pooling |
| `nn.AvgPool2d(k)` | Average pooling |
| `nn.BatchNorm1d(n)` | Batch normalization |
| `nn.Dropout(p)` | Dropout |
| `nn.Flatten()` | Reshape to (N, -1) |
| `nn.Sequential(*layers)` | Layer stack |
| `nn.Embedding(vocab, dim)` | Token embeddings |
| `nn.LSTM(in, hidden)` | LSTM |

### Activations
`nn.ReLU` · `nn.LeakyReLU` · `nn.Sigmoid` · `nn.Tanh` · `nn.Softmax` · `nn.GELU`

### Losses
`nn.cross_entropy` · `nn.mse_loss` · `nn.binary_cross_entropy`

### Optimizers
`optim.SGD` · `optim.Adam` · `optim.AdamW`

### Schedulers
`optim.StepLR` · `optim.CosineAnnealingLR` · `optim.ReduceLROnPlateau` · `optim.LinearWarmup`

### Model API
```python
model.fit(data, y, epochs, lr, batch_size, optimizer, scheduler, val_data, verbose)
model.predict(X)
model.predict_classes(X)
model.accuracy(X, y)
model.save("weights.npz")
model.load("weights.npz")
model.train()
model.eval()
```

---

## How it works

Kiri auto-detects hardware on import and dispatches to the right backend:

- **Apple Silicon (M1/M2/M3/M4)** → [MLX](https://github.com/ml-explore/mlx) backend. Metal GPU + unified memory. Zero-copy CPU↔GPU. Up to 192GB shared memory on M4 Ultra — run 70B models locally.
- **Everything else** → NumPy backend with a built-in autograd engine. No dependencies beyond NumPy.

The same model code runs on both — you write it once, Kiri handles the rest.

---

## Architecture

```
kiri/
├── __init__.py          ← auto-detects hardware, prints report
├── autograd.py          ← autograd engine (CPU backend)
├── model.py             ← Model base class
├── data.py              ← DataLoader, Dataset, TensorDataset
├── nn/
│   ├── layers.py        ← Linear, Conv2d, BatchNorm, Dropout, Sequential
│   ├── activations.py   ← ReLU, Sigmoid, Softmax, GELU, ...
│   ├── pooling.py       ← MaxPool2d, AvgPool2d
│   ├── recurrent.py     ← Embedding, LSTM
│   └── loss.py          ← cross_entropy, mse_loss, bce
├── optim/
│   ├── optimizers.py    ← SGD, Adam, AdamW
│   └── schedulers.py    ← StepLR, CosineAnnealing, ReduceLROnPlateau
└── backend/
    ├── detect.py        ← hardware detection
    └── cpu_ops.py       ← NumPy conv2d, pooling kernels
```

---

## Roadmap

- [x] Dense layers, CNNs, RNNs
- [x] Adam, SGD, AdamW
- [x] LR schedulers
- [x] DataLoader
- [x] Apple Silicon auto-detection
- [ ] Direct Apple Neural Engine dispatch (ANE)
- [ ] Operator fusion (Linear → BN → ReLU in one kernel)
- [ ] ONNX export
- [ ] `kiri.datasets` (MNIST, CIFAR-10 auto-download)

---

## Contributing

PRs welcome. Run tests with:

```bash
pip install kiri-ml[dev]
pytest tests/ -v
```

---

## License

MIT

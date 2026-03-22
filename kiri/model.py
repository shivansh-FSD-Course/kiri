"""
kiri/model.py

Model base class. Key design:
  - MLX backend: uses mlx.nn.value_and_grad correctly
  - CPU backend: manual backward pass
  - .fit() accepts both numpy arrays and DataLoaders
  - optimizer receives the MODEL on MLX, params list on CPU
"""

import numpy as np
from kiri.backend.detect import BACKEND
from kiri.nn.layers import Module
from kiri.nn.loss   import cross_entropy, mse_loss
from kiri.data      import DataLoader

if BACKEND == "mlx":
    import mlx.core as mx
    import mlx.nn as mlx_nn


class Model(Module):
    """
    Base class for Kiri models. Subclass and implement forward().

    Example
    -------
    class MyNet(kiri.Model):
        def __init__(self):
            super().__init__()
            self.fc = kiri.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(x)

    model = MyNet()
    model.fit(X_train, y_train, epochs=10)
    """

    def fit(self, data, y=None, epochs=10, lr=1e-3, batch_size=32,
            loss_fn=None, optimizer=None, verbose=True,
            val_data=None, scheduler=None):
        """
        Train the model.

        Parameters
        ----------
        data       : np.ndarray OR kiri.DataLoader
        y          : labels (only when data is ndarray)
        epochs     : training epochs
        lr         : learning rate (used if optimizer not provided)
        batch_size : mini-batch size (used if data is ndarray)
        loss_fn    : loss function — auto-detected from label dtype if None
        optimizer  : kiri.optim instance
                     On MLX  → pass the MODEL:   kiri.optim.Adam(model, lr=1e-3)
                     On CPU  → pass params list:  kiri.optim.Adam(model.parameters(), lr=1e-3)
        verbose    : print per-epoch summary
        val_data   : (X_val, y_val) or DataLoader
        scheduler  : LR scheduler
        """
        from kiri.optim import Adam

        # Build loader
        if isinstance(data, DataLoader):
            loader = data
            sample_X, sample_y = next(iter(loader))
            if loss_fn is None:
                loss_fn = cross_entropy if np.issubdtype(sample_y.dtype, np.integer) else mse_loss
        else:
            if loss_fn is None:
                loss_fn = cross_entropy if (
                    np.issubdtype(y.dtype, np.integer) or
                    (y.ndim == 1 and y.dtype != np.float32)
                ) else mse_loss
            loader = DataLoader((data, y), batch_size=batch_size, shuffle=True)

        # Build optimizer
        if optimizer is None:
            optimizer = Adam(self, lr=lr) if BACKEND == "mlx" else Adam(self.parameters(), lr=lr)

        history = {"loss": [], "val_loss": []}
        ep_w    = len(str(epochs))

        for epoch in range(epochs):
            self.train()
            epoch_loss, n = 0.0, 0

            for Xb, yb in loader:
                if BACKEND == "mlx":
                    loss_val = self._step_mlx(Xb, yb, loss_fn, optimizer)
                else:
                    loss_val = self._step_cpu(Xb, yb, loss_fn, optimizer)
                epoch_loss += loss_val
                n += 1

            avg = epoch_loss / max(n, 1)
            history["loss"].append(avg)

            if scheduler is not None:
                import inspect
                if "metric" in inspect.signature(scheduler.step).parameters:
                    scheduler.step(avg)
                else:
                    scheduler.step()

            log = {f"loss": avg}

            if val_data is not None:
                self.eval()
                vl = self._eval_loss(val_data, loss_fn)
                history["val_loss"].append(vl)
                log["val_loss"] = vl

            if verbose:
                parts = [f"Epoch {epoch+1:>{ep_w}}/{epochs}"]
                parts += [f"{k}: \033[1m{v:.4f}\033[0m" for k, v in log.items()]
                parts.append(f"lr: {optimizer.lr:.2e}")
                print("  " + "  │  ".join(parts))

        self.train()
        return history

    #training steps

    def _step_mlx(self, Xb, yb, loss_fn, optimizer):
        x_t = mx.array(Xb.astype(np.float32))
        y_t = mx.array(yb)

        def loss_fn_wrap(model, x, y):
            return loss_fn(model.forward(x), y)

        loss_val, grads = mlx_nn.value_and_grad(self, loss_fn_wrap)(self, x_t, y_t)
        optimizer.step(grads)
        mx.eval(self.parameters())
        return float(loss_val)

    def _step_cpu(self, Xb, yb, loss_fn, optimizer):
        from kiri.autograd import Tensor
        optimizer.zero_grad()
        x_t  = Tensor(Xb.astype(np.float32), requires_grad=False)
        pred = self.forward(x_t)
        loss = loss_fn(pred, yb)
        loss.backward()
        optimizer.step()
        return loss.item()

    def _eval_loss(self, val_data, loss_fn):
        if isinstance(val_data, DataLoader):
            losses = []
            for Xv, yv in val_data:
                losses.append(self._eval_batch(Xv, yv, loss_fn))
            return float(np.mean(losses))
        Xv, yv = val_data
        return self._eval_batch(Xv, yv, loss_fn)

    def _eval_batch(self, X, y, loss_fn):
        if BACKEND == "mlx":
            loss = loss_fn(self.forward(mx.array(X.astype(np.float32))), mx.array(y))
            mx.eval(loss)
            return float(loss)
        from kiri.autograd import Tensor
        pred = self.forward(Tensor(X.astype(np.float32), requires_grad=False))
        return loss_fn(pred, y).item()

    # Inference

    def predict(self, X, batch_size=256):
        """Raw model output as np.ndarray."""
        self.eval()
        results = []
        for i in range(0, len(X), batch_size):
            Xb = X[i:i+batch_size].astype(np.float32)
            if BACKEND == "mlx":
                out = self.forward(mx.array(Xb))
                mx.eval(out)
                results.append(np.array(out))
            else:
                from kiri.autograd import Tensor
                results.append(self.forward(Tensor(Xb, requires_grad=False)).data)
        self.train()
        return np.concatenate(results, axis=0)

    def predict_classes(self, X, batch_size=256):
        return np.argmax(self.predict(X, batch_size), axis=-1)

    def accuracy(self, X, y, batch_size=256):
        return float((self.predict_classes(X, batch_size) == y).mean())

    # persistence

    def save(self, path):
        params = self.parameters()
        weights = {}
        for i, p in enumerate(params):
            weights[f"p{i}"] = np.array(p) if BACKEND == "mlx" else p.data
        np.savez(path, **weights)
        print(f"  Saved to {path}")

    def load(self, path):
        data   = np.load(path)
        params = self.parameters()
        for i, p in enumerate(params):
            key = f"p{i}"
            if key not in data:
                continue
            if BACKEND == "mlx":
                p[...] = mx.array(data[key])
            else:
                p.data = data[key].astype(np.float32)
        print(f"  Loaded from {path}")

"""
tests/test_kiri.py

Run with:  pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

# Force CPU backend for tests (no MLX needed)
import kiri.backend.detect as _bd
_bd.BACKEND = "numpy"

from kiri.autograd        import Tensor, _wrap
from kiri.nn.layers       import Linear, Conv2d, Dropout, Flatten, Sequential, BatchNorm1d
from kiri.nn.activations  import ReLU, LeakyReLU, Sigmoid, Tanh, Softmax, GELU
from kiri.nn.pooling      import MaxPool2d, AvgPool2d
from kiri.nn.recurrent    import Embedding, LSTM
from kiri.nn.loss         import cross_entropy, mse_loss, binary_cross_entropy
from kiri.optim           import SGD, Adam, AdamW, StepLR, CosineAnnealingLR, ReduceLROnPlateau
from kiri.data            import DataLoader, TensorDataset
from kiri.model           import Model


# ─── Autograd ─────────────────────────────────────────────────────────────────

class TestAutograd:
    def test_add(self):
        a = Tensor([1.0, 2.0])
        b = Tensor([3.0, 4.0])
        c = (a + b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad, [1, 1])
        np.testing.assert_allclose(b.grad, [1, 1])

    def test_mul(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, 5.0])
        c = (a * b).sum()
        c.backward()
        np.testing.assert_allclose(a.grad, [4, 5])
        np.testing.assert_allclose(b.grad, [2, 3])

    def test_matmul(self):
        a = Tensor(np.ones((3, 4), dtype=np.float32))
        b = Tensor(np.ones((4, 2), dtype=np.float32))
        c = a.matmul(b).sum()
        c.backward()
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4, 2)

    def test_relu(self):
        a = Tensor([-1.0, 0.0, 2.0])
        b = a.relu().sum()
        b.backward()
        np.testing.assert_allclose(a.grad, [0, 0, 1])

    def test_sigmoid(self):
        a = Tensor([0.0])
        b = a.sigmoid()
        b.backward()
        np.testing.assert_allclose(a.grad, [0.25], atol=1e-5)

    def test_mean(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        a.mean().backward()
        np.testing.assert_allclose(a.grad, [[0.25]*2]*2)

    def test_broadcast(self):
        a = Tensor(np.ones((3, 4), dtype=np.float32))
        b = Tensor(np.ones((4,), dtype=np.float32))
        c = (a + b).sum()
        c.backward()
        assert a.grad.shape == (3, 4)
        assert b.grad.shape == (4,)
        np.testing.assert_allclose(b.grad, [3, 3, 3, 3])

    def test_chain_rule(self):
        x = Tensor([2.0])
        # f(x) = relu(x^2 - 3)
        y = (x * x - _wrap(3.0)).relu().sum()
        y.backward()
        # df/dx = 2x * (x^2-3 > 0) = 4
        np.testing.assert_allclose(x.grad, [4.0])

    def test_reshape(self):
        a = Tensor(np.ones((2, 3), dtype=np.float32))
        b = a.reshape(6).sum()
        b.backward()
        assert a.grad.shape == (2, 3)


# ─── Layers ───────────────────────────────────────────────────────────────────

class TestLayers:
    def test_linear_shape(self):
        fc  = Linear(8, 4)
        x   = Tensor(np.random.randn(5, 8).astype(np.float32))
        out = fc(x)
        assert out.shape == (5, 4)

    def test_linear_backward(self):
        fc  = Linear(4, 3, bias=False)
        x   = Tensor(np.random.randn(2, 4).astype(np.float32))
        out = fc(x).sum()
        out.backward()
        assert fc.weight.grad.shape == (3, 4)

    def test_conv2d_shape(self):
        conv = Conv2d(1, 8, kernel_size=3, padding=1)
        x    = Tensor(np.random.randn(2, 1, 8, 8).astype(np.float32))
        out  = conv(x)
        assert out.shape == (2, 8, 8, 8)

    def test_sequential(self):
        net = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x   = Tensor(np.random.randn(3, 4).astype(np.float32))
        assert net(x).shape == (3, 2)

    def test_flatten(self):
        x   = Tensor(np.random.randn(4, 3, 5, 5).astype(np.float32))
        out = Flatten()(x)
        assert out.shape == (4, 75)

    def test_dropout_train(self):
        d = Dropout(0.5); d._is_training = True
        x = Tensor(np.ones((100, 100), dtype=np.float32))
        y = d(x)
        # p=0.5 rescales survivors by 2, so sum ≈ 10000
        # correct check: roughly half the values should be exactly 0
        zeros = (y.data == 0).sum()
        assert 3000 < zeros < 7000  # expect ~50% zeros ± generous tolerance

    def test_dropout_eval(self):
        d = Dropout(0.5); d._is_training = False
        x = Tensor(np.ones((10, 10), dtype=np.float32))
        y = d(x)
        np.testing.assert_array_equal(x.data, y.data)

    def test_batchnorm(self):
        bn  = BatchNorm1d(4)
        x   = Tensor(np.random.randn(8, 4).astype(np.float32))
        out = bn(x)
        assert out.shape == (8, 4)

    def test_maxpool_shape(self):
        x   = Tensor(np.random.randn(2, 4, 8, 8).astype(np.float32))
        out = MaxPool2d(2)(x)
        assert out.shape == (2, 4, 4, 4)


# ─── Activations ─────────────────────────────────────────────────────────────

class TestActivations:
    def _check(self, layer, x_val, expected_out, expected_grad):
        x      = Tensor([x_val])
        result = layer(x)
        s      = result.sum()
        s.backward()
        np.testing.assert_allclose(result.data, [expected_out], atol=1e-4)
        np.testing.assert_allclose(x.grad,      [expected_grad], atol=1e-4)

    def test_relu_pos(self):  self._check(ReLU(),     2.0,  2.0, 1.0)
    def test_relu_neg(self):  self._check(ReLU(),    -1.0,  0.0, 0.0)
    def test_sigmoid(self):   self._check(Sigmoid(),  0.0,  0.5, 0.25)
    def test_tanh(self):      self._check(Tanh(),     0.0,  0.0, 1.0)

    def test_softmax_sums_to_one(self):
        x   = Tensor(np.random.randn(4, 5).astype(np.float32))
        out = Softmax(dim=1)(x)
        np.testing.assert_allclose(out.data.sum(axis=1), np.ones(4), atol=1e-5)


# ─── Losses ───────────────────────────────────────────────────────────────────

class TestLosses:
    def test_cross_entropy_shape(self):
        logits  = Tensor(np.random.randn(8, 4).astype(np.float32))
        targets = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int32)
        loss    = cross_entropy(logits, targets)
        assert loss.data.shape == ()
        assert loss.item() > 0

    def test_cross_entropy_backward(self):
        logits  = Tensor(np.random.randn(4, 3).astype(np.float32))
        targets = np.array([0, 1, 2, 0], dtype=np.int32)
        loss    = cross_entropy(logits, targets)
        loss.backward()
        assert logits.grad.shape == (4, 3)

    def test_mse_loss(self):
        pred   = Tensor(np.array([1.0, 2.0, 3.0]))
        target = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        loss   = mse_loss(pred, target)
        np.testing.assert_allclose(loss.item(), 0.0, atol=1e-6)

    def test_mse_backward(self):
        pred   = Tensor(np.array([2.0, 0.0]))
        target = np.array([1.0, 1.0], dtype=np.float32)
        loss   = mse_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


# ─── Recurrent / Embedding ────────────────────────────────────────────────────

class TestRecurrent:
    def test_embedding_shape(self):
        emb = Embedding(50, 16)
        idx = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
        out = emb(idx)
        assert out.shape == (2, 3, 16)

    def test_embedding_backward(self):
        emb = Embedding(10, 8)
        idx = np.array([[0, 1, 2]], dtype=np.int32)
        out = emb(idx).sum()
        out.backward()
        assert emb.weight.grad.shape == (10, 8)
        # rows 0,1,2 should have non-zero grad; others zero
        assert emb.weight.grad[0].sum() != 0
        assert emb.weight.grad[9].sum() == 0

    def test_lstm_shape(self):
        lstm = LSTM(16, 32)
        x    = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
        out, (h, c) = lstm(x)
        assert out.shape == (2, 5, 32)
        assert h.shape   == (2, 32)
        assert c.shape   == (2, 32)


# ─── Optimizers ───────────────────────────────────────────────────────────────

class TestOptimizers:
    def _make_param(self):
        p      = Tensor(np.array([2.0, -1.0]))
        p.grad = np.array([1.0, -1.0], dtype=np.float32)
        return p

    def test_sgd_step(self):
        p   = self._make_param()
        opt = SGD([p], lr=0.1)
        opt.step()
        np.testing.assert_allclose(p.data, [1.9, -0.9], atol=1e-6)

    def test_adam_step(self):
        p    = self._make_param()
        opt  = Adam([p], lr=0.1)
        prev = p.data.copy()
        opt.step()
        assert not np.allclose(p.data, prev)

    def test_adamw_step(self):
        p   = self._make_param()
        opt = AdamW([p], lr=0.1, weight_decay=0.01)
        opt.step()
        assert p.data is not None

    def test_sgd_zero_grad(self):
        p      = self._make_param()
        opt    = SGD([p], lr=0.1)
        opt.zero_grad()
        np.testing.assert_array_equal(p.grad, [0, 0])


# ─── Schedulers ───────────────────────────────────────────────────────────────

class TestSchedulers:
    def _opt(self, lr=0.1):
        p = Tensor(np.array([1.0]))
        return Adam([p], lr=lr)

    def test_steplr(self):
        opt   = self._opt(0.1)
        sched = StepLR(opt, step_size=2, gamma=0.5)
        sched.step(); sched.step(); sched.step()  # epoch 2 → decay
        np.testing.assert_allclose(opt.lr, 0.05, atol=1e-7)

    def test_cosine(self):
        opt   = self._opt(0.1)
        sched = CosineAnnealingLR(opt, T_max=10, eta_min=0)
        for _ in range(6): sched.step()  # midpoint
        np.testing.assert_allclose(opt.lr, 0.05, atol=1e-3)

    def test_reduce_on_plateau(self):
        opt   = self._opt(0.1)
        sched = ReduceLROnPlateau(opt, patience=2, factor=0.5, verbose=False)
        for _ in range(4): sched.step(1.0)  # no improvement
        assert opt.lr < 0.1


# ─── DataLoader ───────────────────────────────────────────────────────────────

class TestDataLoader:
    def test_length(self):
        X = np.ones((100, 4))
        y = np.ones(100)
        loader = DataLoader((X, y), batch_size=16)
        assert len(loader) == 7  # ceil(100/16)

    def test_batch_shapes(self):
        X = np.random.randn(50, 4).astype(np.float32)
        y = np.random.randint(0, 3, 50).astype(np.int32)
        loader = DataLoader((X, y), batch_size=10, shuffle=False)
        batches = list(loader)
        assert len(batches) == 5
        Xb, yb = batches[0]
        assert Xb.shape == (10, 4)
        assert yb.shape == (10,)

    def test_drop_last(self):
        X = np.ones((105, 4))
        y = np.ones(105)
        loader = DataLoader((X, y), batch_size=10, drop_last=True)
        assert len(loader) == 10

    def test_shuffle(self):
        X = np.arange(100).reshape(100, 1).astype(np.float32)
        y = np.arange(100).astype(np.float32)
        loader1 = DataLoader((X, y), batch_size=100, shuffle=True)
        loader2 = DataLoader((X, y), batch_size=100, shuffle=True)
        b1 = list(loader1)[0][0]
        b2 = list(loader2)[0][0]
        # Two shuffled batches should differ (astronomically unlikely to match)
        assert not np.array_equal(b1, b2)


# ─── End-to-end training ──────────────────────────────────────────────────────

class TestEndToEnd:
    def _make_data(self, n=200, d=8, c=3):
        X = np.random.randn(n, d).astype(np.float32)
        y = np.random.randint(0, c, n).astype(np.int32)
        return X, y

    def _make_model(self):
        class Net(Model):
            def __init__(self):
                super().__init__()
                self.fc1 = Linear(8, 32)
                self.act = ReLU()
                self.fc2 = Linear(32, 3)
            def forward(self, x): return self.fc2(self.act(self.fc1(x)))
        return Net()

    def test_fit_numpy(self):
        X, y  = self._make_data()
        model = self._make_model()
        hist  = model.fit(X, y, epochs=3, lr=1e-3, verbose=False)
        assert len(hist["loss"]) == 3
        assert hist["loss"][-1] < hist["loss"][0]  # loss decreased

    def test_fit_dataloader(self):
        X, y   = self._make_data()
        loader = DataLoader((X, y), batch_size=32)
        model  = self._make_model()
        hist   = model.fit(loader, epochs=3, lr=1e-3, verbose=False)
        assert len(hist["loss"]) == 3

    def test_fit_val_data(self):
        X, y   = self._make_data()
        model  = self._make_model()
        hist   = model.fit(X[:160], y[:160], epochs=3, lr=1e-3,
                           val_data=(X[160:], y[160:]), verbose=False)
        assert "val_loss" in hist
        assert len(hist["val_loss"]) == 3

    def test_fit_with_scheduler(self):
        X, y   = self._make_data()
        model  = self._make_model()
        opt    = Adam(model.parameters(), lr=0.01)
        sched  = StepLR(opt, step_size=2, gamma=0.5)
        model.fit(X, y, epochs=4, optimizer=opt,
                  scheduler=sched, verbose=False)
        # last_epoch goes -1 → 0,1,2,3 after 4 steps
        # lr = 0.01 * 0.5^(3//2) = 0.01 * 0.5^1 = 0.005
        np.testing.assert_allclose(opt.lr, 0.005, atol=1e-7)

    def test_predict_shape(self):
        X, y  = self._make_data()
        model = self._make_model()
        model.fit(X, y, epochs=1, verbose=False)
        preds = model.predict(X)
        assert preds.shape == (200, 3)

    def test_predict_classes(self):
        X, y  = self._make_data()
        model = self._make_model()
        model.fit(X, y, epochs=1, verbose=False)
        classes = model.predict_classes(X)
        assert classes.shape == (200,)
        assert set(classes).issubset({0, 1, 2})

    def test_accuracy(self):
        X, y  = self._make_data()
        model = self._make_model()
        model.fit(X, y, epochs=5, verbose=False)
        acc   = model.accuracy(X, y)
        assert 0.0 <= acc <= 1.0

    def test_save_load(self, tmp_path):
        X, y  = self._make_data()
        model = self._make_model()
        model.fit(X, y, epochs=2, verbose=False)
        path  = str(tmp_path / "model.npz")
        model.save(path)
        preds_before = model.predict(X[:10]).copy()

        # Corrupt weights then reload
        for p in model.parameters(): p.data *= 0
        model.load(path)
        preds_after = model.predict(X[:10])
        np.testing.assert_allclose(preds_before, preds_after, atol=1e-5)

    def test_loss_decreases_mlp(self):
        """Training should consistently decrease loss over 20 epochs."""
        X = np.random.randn(500, 16).astype(np.float32)
        y = np.random.randint(0, 4, 500).astype(np.int32)
        class BigNet(Model):
            def __init__(self):
                super().__init__()
                self.net = Sequential(
                    Linear(16, 64), ReLU(),
                    Linear(64, 32), ReLU(),
                    Linear(32, 4)
                )
            def forward(self, x): return self.net(x)
        model = BigNet()
        hist  = model.fit(X, y, epochs=20, lr=1e-3, verbose=False)
        assert hist["loss"][-1] < hist["loss"][0]

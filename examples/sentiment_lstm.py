"""
examples/sentiment_lstm.py

Text sentiment classification with Embedding + LSTM.
The kind of NLP assignment that used to require Colab.

  python examples/sentiment_lstm.py
"""

import numpy as np
import kiri
import kiri.nn as nn


# ── tiny vocab + fake data ────────────────────────────────────────────────────

VOCAB_SIZE  = 1000
MAX_LEN     = 20
EMBED_DIM   = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 2   # positive / negative

rng = np.random.default_rng(42)
X_train = rng.integers(0, VOCAB_SIZE, (800, MAX_LEN), dtype=np.int32)
y_train = rng.integers(0, NUM_CLASSES, 800, dtype=np.int32)
X_test  = rng.integers(0, VOCAB_SIZE, (200, MAX_LEN), dtype=np.int32)
y_test  = rng.integers(0, NUM_CLASSES, 200, dtype=np.int32)


# ── model ─────────────────────────────────────────────────────────────────────

class SentimentLSTM(kiri.Model):
    def __init__(self):
        self.embed   = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.lstm    = nn.LSTM(EMBED_DIM, HIDDEN_SIZE)
        self.dropout = nn.Dropout(0.3)
        self.fc      = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        # x: (N, seq_len) integer tokens
        embedded = self.embed(x)                    # (N, seq_len, EMBED_DIM)
        out, (h_n, _) = self.lstm(embedded)         # h_n: (N, HIDDEN_SIZE)
        h_n = self.dropout(h_n)
        return self.fc(h_n)


# ── train ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model     = SentimentLSTM()
    optimizer = kiri.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = kiri.optim.CosineAnnealingLR(optimizer, T_max=10)

    loader     = kiri.DataLoader((X_train, y_train), batch_size=32, shuffle=True)
    val_loader = kiri.DataLoader((X_test,  y_test),  batch_size=64, shuffle=False)

    history = model.fit(
        loader,
        epochs=10,
        optimizer=optimizer,
        scheduler=scheduler,
        val_data=val_loader,
        verbose=True,
    )

    acc = model.accuracy(X_test, y_test)
    print(f"\n✓ Test accuracy: {acc*100:.1f}%")

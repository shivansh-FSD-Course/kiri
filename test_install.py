import kiri
import kiri.nn as nn
import numpy as np

class Net(kiri.Model):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, 10)
        )
    def forward(self, x):
        return self.net(x)

X = np.random.randn(1000, 512).astype(np.float32)
y = np.random.randint(0, 10, 1000).astype(np.int32)

model = Net()
model.fit(X, y, epochs=3, lr=1e-3, verbose=True)
print("✓ install from PyPI works")
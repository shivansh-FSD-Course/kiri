"""
 ██╗  ██╗██╗██████╗ ██╗
 ██║ ██╔╝██║██╔══██╗██║
 █████╔╝ ██║██████╔╝██║
 ██╔═██╗ ██║██╔══██╗██║
 ██║  ██╗██║██║  ██║██║
 ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝

Lightweight ML for everyone. No CUDA required.
"""

__version__ = "0.2.0"

from kiri.backend.detect     import BACKEND, get_hardware_report
from kiri.backend.accelerate import is_available as accelerate_available
from kiri.model              import Model
from kiri.data               import DataLoader, Dataset, TensorDataset
from kiri.benchmark          import benchmark
from kiri.backend.fusion     import fuse_model
import kiri.nn    as nn
import kiri.optim as optim

print(get_hardware_report())
_accel = "✓ Accelerate/AMX" if accelerate_available() else "✗ Accelerate"
print(f"  Kiri v{__version__}  │  backend: {BACKEND}  │  {_accel}\n")

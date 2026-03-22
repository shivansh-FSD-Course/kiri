"""
 ██╗  ██╗██╗██████╗ ██╗
 ██║ ██╔╝██║██╔══██╗██║
 █████╔╝ ██║██████╔╝██║
 ██╔═██╗ ██║██╔══██╗██║
 ██║  ██╗██║██║  ██║██║
 ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝

Lightweight ML for everyone. No CUDA required.
"""

__version__ = "0.3.0"

from kiri.backend.detect     import BACKEND, get_hardware_report
from kiri.backend.accelerate import is_available as accelerate_available
from kiri.backend.ane        import is_available as ane_available, ane_optimize
from kiri.model              import Model
from kiri.data               import DataLoader, Dataset, TensorDataset
from kiri.benchmark          import benchmark
from kiri.backend.fusion     import fuse_model
import kiri.nn    as nn
import kiri.optim as optim

print(get_hardware_report())
_accel = "✓ Accelerate/AMX" if accelerate_available() else "✗ Accelerate"
_ane   = "✓ ANE/CoreML"     if ane_available()         else "✗ ANE"
print(f"  Kiri v{__version__}  │  backend: {BACKEND}  │  {_accel}  │  {_ane}\n")

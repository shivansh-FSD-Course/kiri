"""
 ██╗  ██╗██╗██████╗ ██╗
 ██║ ██╔╝██║██╔══██╗██║
 █████╔╝ ██║██████╔╝██║
 ██╔═██╗ ██║██╔══██╗██║
 ██║  ██╗██║██║  ██║██║
 ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝

Lightweight ML for everyone. No CUDA required.
"""

__version__ = "0.1.0"

from kiri.backend.detect import BACKEND, get_hardware_report
from kiri.model          import Model
from kiri.data           import DataLoader, Dataset, TensorDataset
import kiri.nn    as nn
import kiri.optim as optim

print(get_hardware_report())
print(f"  Kiri v{__version__}  │  backend: {BACKEND}\n")

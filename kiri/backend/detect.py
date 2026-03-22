"""
kiri/backend/detect.py

Auto-detects the best available backend on import.
  Apple Silicon + MLX installed  →  "mlx"
  Everything else                →  "numpy"
"""

import platform
import subprocess


def detect_backend():
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            import mlx.core  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "numpy"


def get_memory_gb():
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True
        ).stdout.strip()
        return int(out) // (1024 ** 3)
    except Exception:
        return 0


def get_hardware_report():
    backend = detect_backend()
    lines = ["╭─ Kiri 🌫️ ─────────────────────────────╮"]
    if backend == "mlx":
        mem = get_memory_gb()
        lines.append(f"│  Backend  : Apple Silicon (MLX)        │")
        lines.append(f"│  Chip     : {platform.machine():<28}│")
        lines.append(f"│  Memory   : {str(mem)+'GB unified memory':<28}│")
        lines.append(f"│  Status   : ✓ Metal GPU + CPU active   │")
    else:
        cpu = (platform.processor() or platform.machine())[:28]
        lines.append(f"│  Backend  : CPU (NumPy)                │")
        lines.append(f"│  CPU      : {cpu:<28}│")
        lines.append(f"│  Status   : ✓ Running on CPU           │")
        lines.append(f"│  Tip      : Use Apple Silicon for 10x  │")
        lines.append(f"│           : faster training            │")
    lines.append("╰────────────────────────────────────────╯")
    return "\n".join(lines)


BACKEND = detect_backend()

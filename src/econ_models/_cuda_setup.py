"""
Auto-configure LD_LIBRARY_PATH for pip-installed NVIDIA/CUDA libraries.

When ``tensorflow[and-cuda]`` is installed, the NVIDIA shared libraries
(cuBLAS, cuDNN, cuSOLVER …) land in ``site-packages/nvidia/*/lib/`` but
TensorFlow cannot find them unless ``LD_LIBRARY_PATH`` includes those
directories.

Calling ``configure()`` **before** the first ``import tensorflow`` adds
every ``nvidia/*/lib/`` directory it can find to ``LD_LIBRARY_PATH``
so GPU support works out of the box—no manual shell patching required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_nvidia_lib_dirs() -> list[str]:
    """Return all ``nvidia/*/lib`` directories found in ``sys.path``."""
    lib_dirs: list[str] = []
    seen: set[str] = set()

    for base in sys.path:
        nvidia_root = Path(base) / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for child in sorted(nvidia_root.iterdir()):
            lib_dir = child / "lib"
            if lib_dir.is_dir():
                resolved = str(lib_dir.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    lib_dirs.append(resolved)
    return lib_dirs


def configure() -> None:
    """Prepend pip-installed NVIDIA lib dirs to ``LD_LIBRARY_PATH``."""
    nvidia_dirs = _find_nvidia_lib_dirs()
    if not nvidia_dirs:
        return  # No pip-installed NVIDIA packages — nothing to do.

    existing = os.environ.get("LD_LIBRARY_PATH", "")
    existing_set = set(existing.split(os.pathsep)) if existing else set()

    new_dirs = [d for d in nvidia_dirs if d not in existing_set]
    if not new_dirs:
        return  # Already configured.

    parts = new_dirs + ([existing] if existing else [])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(parts)

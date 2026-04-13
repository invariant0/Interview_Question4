#!/usr/bin/env bash
# ---------------------------------------------------------------
# setup_cuda_env.sh
#
# Patches the venv activate script so that LD_LIBRARY_PATH
# includes the pip-installed NVIDIA/CUDA shared libraries
# shipped with tensorflow[and-cuda].
#
# Usage:  source setup_cuda_env.sh
#   (run once after creating the venv and pip install -e .)
# ---------------------------------------------------------------

set -euo pipefail

VENV_DIR="${VIRTUAL_ENV:?Activate your venv first (source tf/bin/activate)}"
ACTIVATE="$VENV_DIR/bin/activate"
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
NVIDIA_DIR="$VENV_DIR/lib/python${PY_VER}/site-packages/nvidia"

if [ ! -d "$NVIDIA_DIR" ]; then
    echo "ERROR: $NVIDIA_DIR not found. Run 'pip install -e .' first."
    exit 1
fi

# Check if the CUDA block is already present
if grep -q 'NVIDIA/CUDA libraries for TensorFlow GPU support' "$ACTIVATE"; then
    echo "CUDA LD_LIBRARY_PATH block already present in activate script."
else
    # Build LD_LIBRARY_PATH entries from all nvidia sub-packages that have a lib/ dir
    LIB_DIRS=""
    for d in "$NVIDIA_DIR"/*/lib; do
        [ -d "$d" ] && LIB_DIRS="${LIB_DIRS:+${LIB_DIRS}:}\$_NVIDIA_PKG_DIR/$(basename "$(dirname "$d")")/lib"
    done

    # Inject the block into activate, right after the PATH export
    CUDA_BLOCK=$(cat <<CUDA_EOF

# --- NVIDIA/CUDA libraries for TensorFlow GPU support ---
_OLD_VIRTUAL_LD_LIBRARY_PATH="\${LD_LIBRARY_PATH:-}"
_NVIDIA_PKG_DIR="\$VIRTUAL_ENV/lib/python${PY_VER}/site-packages/nvidia"
LD_LIBRARY_PATH="${LIB_DIRS}\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
unset _NVIDIA_PKG_DIR
CUDA_EOF
)

    # Also inject the restore block into deactivate()
    DEACTIVATE_BLOCK='    # Restore LD_LIBRARY_PATH\
    if [ -n "${_OLD_VIRTUAL_LD_LIBRARY_PATH+x}" ] ; then\
        LD_LIBRARY_PATH="${_OLD_VIRTUAL_LD_LIBRARY_PATH:-}"\
        export LD_LIBRARY_PATH\
        unset _OLD_VIRTUAL_LD_LIBRARY_PATH\
    fi'

    # Check if deactivate restore block already exists
    if ! grep -q '_OLD_VIRTUAL_LD_LIBRARY_PATH' "$ACTIVATE"; then
        # Insert LD_LIBRARY_PATH restore into deactivate(), after the PYTHONHOME restore block
        sed -i '/unset _OLD_VIRTUAL_PYTHONHOME/a\
\
'"$DEACTIVATE_BLOCK" "$ACTIVATE"
    fi

    # Append CUDA block after "export PATH"
    sed -i '/^export PATH$/a\'"$CUDA_BLOCK" "$ACTIVATE"

    echo "Patched $ACTIVATE with CUDA LD_LIBRARY_PATH."
    echo "Run 'deactivate && source tf/bin/activate' to pick up changes."
fi

# Auto-configure LD_LIBRARY_PATH for pip-installed NVIDIA libs
# so TensorFlow finds CUDA without manual shell patching.
from econ_models._cuda_setup import configure as _configure_cuda
_configure_cuda()
del _configure_cuda

__version__ = "0.1.0"
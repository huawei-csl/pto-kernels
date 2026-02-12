# Import torch is required to avoid "libc10.so: cannot open shared object file: No such file or directory"
# See https://github.com/facebookresearch/pytorch3d/issues/1531#issuecomment-1538198217
import torch  # noqa

from .pto_kernels_ops import *  # noqa

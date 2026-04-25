# Import torch before the extension when available to avoid missing libc10
# loader errors in built custom-op environments.
try:
    import torch  # noqa

    from .pto_kernels_ops import *  # noqa
except (ImportError, OSError) as exc:
    EXTENSION_IMPORT_ERROR = exc
else:
    EXTENSION_IMPORT_ERROR = None

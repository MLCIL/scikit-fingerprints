import numpy as np
from scipy.sparse import csr_array

"""
Bulk metrics compute intersections as a Gram matrix X @ X.T

It uses the same dtype as input fingerprint: uint8 (binary) or uint32 (counts).
Those can overflow for denser fingerprints (e.g. RDKit), so we need to make sure
dtype is large enough.
"""


def array_to_binary_csr(X: np.ndarray | csr_array | list) -> csr_array:
    """
    Convert binary array to sparse uint32 CSR.
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    return X.astype(np.promote_types(X.dtype, np.uint32), copy=False)


def array_to_count_csr(X: np.ndarray | csr_array | list) -> csr_array:
    """
    Convert count array to sparse int64 CSR.
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    return X.astype(np.promote_types(X.dtype, np.int64), copy=False)

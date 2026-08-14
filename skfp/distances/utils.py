import numpy as np
from scipy.sparse import csr_array


def array_to_csr(X: np.ndarray | csr_array | list) -> csr_array:
    """
    Convert array to sparse uint32 CSR.

    Bulk metrics compute intersections as a Gram matrix X @ X.T. This uses the same
    dtype as input fingerprint, so for binary fingerprints using uint8 this can
    overflow. Casting to uint32 fixes this.
    """
    if not isinstance(X, csr_array):
        X = csr_array(X)

    return X.astype(np.promote_types(X.dtype, np.uint32), copy=False)

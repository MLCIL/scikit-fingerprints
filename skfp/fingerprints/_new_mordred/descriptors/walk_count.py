import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    # Molecular Walk Count
    "MWC01",
    "MWC02",
    "MWC03",
    "MWC04",
    "MWC05",
    "MWC06",
    "MWC07",
    "MWC08",
    "MWC09",
    "MWC10",
    # Total MWC
    "TMWC10",
    # Self-Returning Walk
    "SRW02",
    "SRW03",
    "SRW04",
    "SRW05",
    "SRW06",
    "SRW07",
    "SRW08",
    "SRW09",
    "SRW10",
    # Total SRW
    "TSRW10",
]


MAX_ORDER = 10


def calc(
    props: AtomicProperties, eigendecomposition: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Walk count descriptors.

    A walk of length k is a sequence of k bonds, and the entries of the k-th power
    of the adjacency matrix count the walks of that length between two atoms. The
    whole matrix sums to the molecular walk count, and its trace counts the walks
    returning to the atom they started from.

    Both sums follow from the eigenvalues of the adjacency matrix, because its
    powers have the same eigenvectors and the k-th power of an eigenvalue. This
    is faster than repeated matrix multiplications.
    """
    eigvals, eigvecs = eigendecomposition

    # the weight each eigenvalue carries in a sum over the whole matrix
    weights = eigvecs.sum(axis=0) ** 2

    orders = np.arange(1, MAX_ORDER + 1)[:, np.newaxis]
    eigval_powers = eigvals**orders

    # walks are counted, so both sums are whole numbers, and rounding to one takes
    # out whatever the eigenvalue powers left behind of their cancellation
    walk_counts = np.rint(eigval_powers @ weights)
    self_returning_counts = np.rint(eigval_powers.sum(axis=1))

    # the first molecular walk count is the bond count, the rest are on a log scale
    molecular = np.concatenate([[0.5 * walk_counts[0]], np.log(walk_counts[1:] + 1)])
    self_returning = np.log(self_returning_counts[1:] + 1)

    values = [
        *molecular,
        props.num_atoms + molecular.sum(),
        *self_returning,
        props.num_atoms + self_returning.sum(),
    ]

    return np.asarray(values, dtype=np.float32)

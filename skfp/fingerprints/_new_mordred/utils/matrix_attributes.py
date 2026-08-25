import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# the attributes in the order the descriptors expose them
ATTRIBUTE_NAMES = [
    "graph_energy",
    "leading_eigenvalue",
    "spectral_diameter",
    "sp_ad",
    "sp_mad",
    "log_ee",
    "sm1",
    "ve1",
    "ve2",
    "ve3",
    "vr1",
    "vr2",
    "vr3",
]


def spectral_attributes(
    matrices: np.ndarray,
    props: AtomicProperties,
    hermitian: bool,
    n_frags: int,
    eigendecomposition: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """
    Compute the spectral attributes of every matrix in a stack, of shape
    ``(n_matrices, 13)`` with the attributes in :data:`ATTRIBUTE_NAMES` order.

    The Barysz descriptors ask for the attributes of eight matrices of the same
    molecule, and taking them together keeps the reductions off arrays of a few
    dozen numbers, where the dispatch costs more than the arithmetic. A single
    matrix is the stack of one, so the formulas live in one place either way.

    Requires a connected molecule (single fragment). For disconnected molecules, all
    attributes propagate NaN.
    """
    num_atoms = props.num_atoms
    if n_frags != 1:
        return np.full((len(matrices), len(ATTRIBUTE_NAMES)), np.nan)

    eigvals, eigvecs = _eigendecompositions(matrices, hermitian, eigendecomposition)
    matrix_idxs = np.arange(len(matrices))
    leading = eigvals[matrix_idxs, eigvals.argmax(axis=1)]
    smallest = eigvals[matrix_idxs, eigvals.argmin(axis=1)]
    # the eigenvector belonging to the largest eigenvalue of each matrix
    leading_eigvecs = eigvecs[matrix_idxs, :, eigvals.argmax(axis=1)]

    mean_eigenvalue = eigvals.mean(axis=1)
    sp_ad = np.abs(eigvals - mean_eigenvalue[:, np.newaxis]).sum(axis=1)
    ve1 = np.abs(leading_eigvecs).sum(axis=1)
    vr1 = _randic_like_index(leading_eigvecs, props)

    return np.stack(
        [
            # graph energy (SpAbs)
            np.abs(eigvals).sum(axis=1),
            # leading eigenvalue (SpMax) and the spread of the spectrum (SpDiam)
            leading,
            leading - smallest,
            # spectral absolute deviation, and its mean over the atoms
            sp_ad,
            sp_ad / num_atoms,
            _log_estrada_index(eigvals, leading),
            # spectral moment
            np.trace(matrices, axis1=1, axis2=2),
            # coefficient sum of the leading eigenvector, its average and logarithm
            ve1,
            ve1 / num_atoms,
            np.log(0.1 * num_atoms * ve1),
            # Randic-like eigenvector index, its average and logarithm
            vr1,
            vr1 / num_atoms,
            np.where(vr1 > 0, np.log(0.1 * num_atoms * vr1), np.nan),
        ],
        axis=1,
    )


class MatrixAttributes:
    """
    Spectral attributes derived from a single graph matrix.

    Requires a connected molecule (single fragment). For disconnected molecules, all
    attributes propagate NaN.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        props: AtomicProperties,
        hermitian: bool,
        n_frags: int,
        eigendecomposition: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        stacked = (
            None
            if eigendecomposition is None
            else (eigendecomposition[0][np.newaxis], eigendecomposition[1][np.newaxis])
        )
        (
            self.graph_energy,
            self.leading_eigenvalue,
            self.spectral_diameter,
            self.sp_ad,
            self.sp_mad,
            self.log_ee,
            self.sm1,
            self.ve1,
            self.ve2,
            self.ve3,
            self.vr1,
            self.vr2,
            self.vr3,
        ) = spectral_attributes(matrix[np.newaxis], props, hermitian, n_frags, stacked)[
            0
        ]


def _eigendecompositions(
    matrices: np.ndarray,
    hermitian: bool,
    known: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalues and eigenvectors of every matrix in the stack, unless the caller has
    already decomposed them for another descriptor.

    The matrices are decomposed one at a time: numpy can take a whole stack in one
    call, but that turned out to be slower here, and it takes a different LAPACK
    path whose last bits differ.
    """
    if known is not None:
        return known

    decompose = np.linalg.eigh if hermitian else np.linalg.eig
    decomposed = [decompose(matrix) for matrix in matrices]
    eigvals = np.stack([np.real(values) for values, _ in decomposed])
    eigvecs = np.stack([np.real(vectors) for _, vectors in decomposed])
    return eigvals, eigvecs


def _log_estrada_index(eigvals: np.ndarray, leading: np.ndarray) -> np.ndarray:
    """
    Estrada-like index of every matrix, ``LogEE = log(sum(exp(lambda_i)))`` over its
    eigenvalues ``lambda_i``.

    Computed via the log-sum-exp trick for numerical stability
    (see https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp):
    ``log(sum(exp(x_i))) = a + log(sum(exp(x_i - a)))`` with ``a = max(x_i)``.

    Note that this intentionally diverges from mordred-community, whose
    implementation adds a spurious ``exp(-a)`` term and thus computes
    ``log(1 + sum(exp(lambda_i)))`` instead of the documented formula.
    See https://github.com/JacksonBurns/mordred-community/issues/24.
    """
    shift = np.maximum(leading, 0)
    shifted_eigvals = eigvals - shift[:, np.newaxis]
    log_sum_exp = np.log(np.sum(np.exp(shifted_eigvals), axis=1))
    return shift + log_sum_exp


@np.errstate(divide="ignore", invalid="ignore")
def _randic_like_index(
    leading_eigvecs: np.ndarray, props: AtomicProperties
) -> np.ndarray:
    """
    Randic-like index of every matrix, over the bonds, weighted by its leading
    eigenvector.
    """
    begins = leading_eigvecs[:, props.bond_begin_idxs]
    ends = leading_eigvecs[:, props.bond_end_idxs]
    return ((begins * ends) ** -0.5).sum(axis=1)

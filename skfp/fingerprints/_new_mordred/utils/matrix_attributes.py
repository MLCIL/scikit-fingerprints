import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class MatrixAttributes:
    """
    Spectral attributes derived from a graph matrix.

    Requires a connected molecule (single fragment). For disconnected molecules, all
    attributes propagate NaN.
    """

    def __init__(
        self, matrix: np.ndarray, props: AtomicProperties, hermitian: bool, n_frags: int
    ):
        num_atoms = props.num_atoms
        eigvals, eigvecs = _eigendecomposition(matrix, hermitian, n_frags, num_atoms)
        i_min, i_max = int(np.argmin(eigvals)), int(np.argmax(eigvals))
        leading_eigvec = eigvecs[:, i_max]

        # graph energy (SpAbs)
        self.graph_energy = np.abs(eigvals).sum()
        # leading eigenvalue (SpMax)
        self.leading_eigenvalue = eigvals[i_max]
        # spectral diameter (SpDiam)
        self.spectral_diameter = self.leading_eigenvalue - eigvals[i_min]
        # mean of the eigenvalues (SpMean)
        self.mean_eigenvalue = np.mean(eigvals)
        # spectral absolute deviation, and its mean over the atoms
        self.sp_ad = np.abs(eigvals - self.mean_eigenvalue).sum()
        self.sp_mad = self.sp_ad / num_atoms
        # spectral moment
        self.sm1 = np.trace(matrix)
        self.log_ee = _log_estrada_index(eigvals, i_max)

        # coefficient sum of the leading eigenvector, its average and its logarithm
        self.ve1 = np.abs(leading_eigvec).sum()
        self.ve2 = self.ve1 / num_atoms
        self.ve3 = np.log(0.1 * num_atoms * self.ve1)

        # Randic-like eigenvector-based index, its average and its logarithm
        self.vr1 = _randic_like_index(leading_eigvec, props)
        self.vr2 = self.vr1 / num_atoms
        self.vr3 = (
            np.log(0.1 * num_atoms * self.vr1) if self.vr1 > 0 else np.float64(np.nan)
        )


def _eigendecomposition(
    matrix: np.ndarray, hermitian: bool, n_frags: int, num_atoms: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigenvalues and eigenvectors of a graph matrix.

    NaN for multi-fragment molecules.
    """
    if n_frags != 1:
        return np.full(num_atoms, np.nan), np.full((num_atoms, num_atoms), np.nan)

    eigvals, eigvecs = np.linalg.eigh(matrix) if hermitian else np.linalg.eig(matrix)
    return np.real(eigvals), np.real(eigvecs)


def _log_estrada_index(eigvals: np.ndarray, i_max: int) -> np.floating:
    """
    Estrada-like index, defined as ``LogEE = log(sum(exp(lambda_i)))`` over the
    eigenvalues ``lambda_i``.

    Computed via the log-sum-exp trick for numerical stability
    (see https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp):
    ``log(sum(exp(x_i))) = a + log(sum(exp(x_i - a)))`` with ``a = max(x_i)``.

    Note that this intentionally diverges from mordred-community, whose
    implementation adds a spurious ``exp(-a)`` term and thus computes
    ``log(1 + sum(exp(lambda_i)))`` instead of the documented formula.
    See https://github.com/JacksonBurns/mordred-community/issues/24.
    """
    a = np.maximum(eigvals[i_max], 0)
    return a + np.log(np.exp(eigvals - a).sum())


@np.errstate(divide="ignore", invalid="ignore")
def _randic_like_index(
    leading_eigvec: np.ndarray, props: AtomicProperties
) -> np.floating:
    """
    Randic-like index over the bonds, weighted by the leading eigenvector.
    """
    begins = leading_eigvec[props.bond_begin_idxs]
    ends = leading_eigvec[props.bond_end_idxs]
    return ((begins * ends) ** -0.5).sum()

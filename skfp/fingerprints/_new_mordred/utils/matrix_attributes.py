from functools import cached_property

import numpy as np
from rdkit.Chem import Mol


class MatrixAttributes:
    """Spectral attributes derived from a graph matrix.

    Requires a connected molecule (single fragment).
    For disconnected molecules, all attributes propagate NaN.
    """

    def __init__(self, matrix: np.ndarray, mol: Mol, hermitian: bool, n_frags: int):
        self._matrix = matrix
        self._mol = mol
        self._n_atoms: int = mol.GetNumAtoms()

        # not connected
        if n_frags != 1:
            n = self._n_atoms
            self._vals = np.full(n, np.nan)
            self._vecs = np.full((n, n), np.nan)
            self._i_min = 0
            self._i_max = 0
            return

        w, v = (np.linalg.eigh if hermitian else np.linalg.eig)(matrix)

        if np.iscomplexobj(w):
            w = w.real
        if np.iscomplexobj(v):
            v = v.real

        self._vals = w
        self._vecs = v
        self._i_min = int(np.argmin(w))
        self._i_max = int(np.argmax(w))

    @cached_property
    def sp_abs(self) -> np.floating:
        """Graph energy."""
        return np.abs(self._vals).sum()

    @cached_property
    def sp_max(self) -> np.floating:
        """Leading eigenvalue."""
        return self._vals[self._i_max]

    @cached_property
    def sp_diam(self) -> np.floating:
        """Spectral diameter."""
        return self.sp_max - self._vals[self._i_min]

    @cached_property
    def sp_mean(self) -> np.floating:
        """Mean of eigenvalues."""
        return np.mean(self._vals)

    @cached_property
    def sp_ad(self) -> np.floating:
        """Spectral absolute deviation."""
        return np.abs(self._vals - self.sp_mean).sum()

    @cached_property
    def sp_mad(self) -> np.floating:
        """Spectral mean absolute deviation."""
        return self.sp_ad / self._n_atoms

    @cached_property
    def log_ee(self) -> np.floating:
        """Estrada-like index (log-sum-exp).

        log sum exp: https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp.
        """
        a = np.maximum(self._vals[self._i_max], 0)
        sx = np.exp(self._vals - a).sum() + np.exp(-a)
        return a + np.log(sx)

    @cached_property
    def sm1(self) -> np.floating:
        """Spectral moment."""
        return np.trace(self._matrix)

    @cached_property
    def ve1(self) -> np.floating:
        """Coefficient sum of the last eigenvector."""
        return np.abs(self._vecs[:, self._i_max]).sum()

    @cached_property
    def ve2(self) -> np.floating:
        """Average coefficient of the last eigenvector."""
        return self.ve1 / self._n_atoms

    @cached_property
    def ve3(self) -> np.floating:
        """Logarithmic coefficient sum of the last eigenvector."""
        return np.log(0.1 * self._n_atoms * self.ve1)

    @cached_property
    def vr1(self) -> float:
        """Randic-like eigenvector-based index."""
        s = 0.0
        for bond in self._mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            s += (self._vecs[i, self._i_max] * self._vecs[j, self._i_max]) ** -0.5
        return s

    @cached_property
    def vr2(self) -> np.floating:
        """Normalized Randic-like eigenvector-based index."""
        return self.vr1 / self._n_atoms

    @cached_property
    def vr3(self) -> np.floating:
        """Logarithmic Randic-like eigenvector-based index."""
        return np.log(0.1 * self._n_atoms * self.vr1)

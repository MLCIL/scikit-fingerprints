from functools import cached_property

import numpy as np
from rdkit.Chem import GetAdjacencyMatrix, GetDistanceMatrix, Mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


class DistanceMatrix:
    hermitian = True

    def __init__(self, mol: Mol, use_bo: bool = False, use_atom_wts: bool = False):
        self.matrix: np.ndarray
        self.matrix = GetDistanceMatrix(
            mol, useBO=use_bo, useAtomWts=use_atom_wts, force=True
        )

    @cached_property
    def eccentricity(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.floating:
        return self.eccentricity().min()

    @cached_property
    def diameter(self) -> np.floating:
        return self.matrix.max()


class AdjacencyMatrix:
    hermitian = True

    def __init__(self, mol: Mol, use_bo: bool = False):
        self._base: np.ndarray
        self._base = GetAdjacencyMatrix(mol, useBO=use_bo, force=True)
        self._orders = [self._base]

    def order(self, n: int = 1) -> np.ndarray:
        while len(self._orders) < n:
            self._orders.append(self._orders[-1].dot(self._base))
        return self._orders[n - 1]

    @cached_property
    def degree(self) -> np.ndarray:
        """Number of edges incident to each vertex (atom).

        By default ``use_bo=False``, so bond orders are ignored and each
        bond counts as one edge. In that case, atom degree equals the atom
        valence, i.e. the number of bonds each atom forms.
        """
        return self._base.sum(axis=0)


class DistanceMatrix3D:
    def __init__(self, coords: np.ndarray, use_atom_wts: bool = False):
        self.matrix: np.ndarray
        self.matrix = np.sqrt(np.sum((coords[:, np.newaxis] - coords) ** 2, axis=2))
        self.use_atom_wts = use_atom_wts

    @cached_property
    def eccentricities(self) -> np.ndarray:
        return self.matrix.max(axis=0)

    @cached_property
    def radius(self) -> np.floating:
        return self.eccentricities().min()

    @cached_property
    def diameter(self) -> np.floating:
        return self.matrix.max()

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol


class DistanceMatrix:
    def __init__(self, mol: Mol, useBO: bool = False, useAtomWts: bool = False):
        self.matrix: np.ndarray
        self.matrix = Chem.GetDistanceMatrix(
            mol, useBO=useBO, useAtomWts=useAtomWts, force=True
        )
        self._eccentricity: np.ndarray | None = None
        self._radius: np.floating | None = None
        self._diameter: np.floating | None = None

    def eccentricity(self) -> np.ndarray:
        if self._eccentricity is None:
            self._eccentricity = self.matrix.max(axis=0)
        return self._eccentricity

    def radius(self) -> np.floating:
        if self._radius is None:
            self._radius = self.eccentricity().min()
        return self._radius

    def diameter(self) -> np.floating:
        if self._diameter is None:
            self._diameter = self.matrix.max()
        return self._diameter


class AdjacencyMatrix:
    def __init__(self, mol: Mol, useBO: bool = False):
        self._base: np.ndarray
        self._base = Chem.GetAdjacencyMatrix(mol, useBO=useBO, force=True)
        self._orders = [self._base]
        self._valence: np.ndarray | None = None

    def order(self, n: int = 1) -> np.ndarray:
        while len(self._orders) < n:
            self._orders.append(self._orders[-1].dot(self._base))
        return self._orders[n - 1]

    def valence(self) -> np.ndarray:
        if self._valence is None:
            self._valence = self._base.sum(axis=0)
        return self._valence


class DistanceMatrix3D:
    def __init__(self, coords: np.ndarray, useAtomWts: bool = False):
        self.matrix: np.ndarray
        self.matrix = np.sqrt(np.sum((coords[:, np.newaxis] - coords) ** 2, axis=2))
        self._eccentricity: np.ndarray | None = None
        self._radius: np.floating | None = None
        self._diameter: np.floating | None = None

    def eccentricity(self) -> np.ndarray:
        if self._eccentricity is None:
            self._eccentricity = self.matrix.max(axis=0)
        return self._eccentricity

    def radius(self) -> np.floating:
        if self._radius is None:
            self._radius = self.eccentricity().min()
        return self._radius

    def diameter(self) -> np.floating:
        if self._diameter is None:
            self._diameter = self.matrix.max()
        return self._diameter

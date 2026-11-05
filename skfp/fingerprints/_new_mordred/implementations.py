from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred import calculator

"""
Implementations of Mordred descriptors, selected with the ``implementation`` parameter
of ``MordredFingerprint``. The ``mordredcommunity`` one is in its own module, since
it imports that optional dependency.
"""


class MordredImplementation(ABC):
    """
    Computes Mordred descriptors of molecules. Each implementation takes the
    ``MordredFingerprint`` parameters it uses.
    """

    @abstractmethod
    def get_feature_names(self) -> np.ndarray:
        """Names of the computed descriptors, in output order."""

    @abstractmethod
    def compute(self, mols: Sequence[Mol]) -> np.ndarray:
        """
        Float32 array of shape (n_mols, n_features) with descriptor values, and NaN
        where they are undefined, e.g. 3D descriptors of molecules without a 3D
        conformer.
        """


class SkfpMordredImplementation(MordredImplementation):
    """Vectorized Mordred descriptors built into scikit-fingerprints."""

    def __init__(self, use_3D: bool, timeout: float, max_ring_system_rank: int):
        self.use_3D = use_3D
        self.timeout = timeout
        self.max_ring_system_rank = max_ring_system_rank

    def get_feature_names(self) -> np.ndarray:
        return calculator.get_feature_names(self.use_3D)

    def compute(self, mols: Sequence[Mol]) -> np.ndarray:
        X = [
            calculator.compute(
                mol=mol,
                use_3D=self.use_3D,
                timeout=self.timeout,
                max_ring_system_rank=self.max_ring_system_rank,
            )
            for mol in mols
        ]
        return np.array(X, dtype=np.float32)

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from rdkit.Chem import Mol

from skfp.fingerprints._mordred_implementations.base import BaseMordredImplementation
from skfp.fingerprints._new_mordred import calculator


class OptimizedMordredImplementation(BaseMordredImplementation):
    """
    Optimized Mordred descriptors, built from scratch in scikit-fingerprints.
    """

    def __init__(self, use_3D: bool, timeout: float, max_ring_system_rank: int):
        self.use_3D = use_3D
        self.timeout = timeout
        self.max_ring_system_rank = max_ring_system_rank

    def get_feature_names(self) -> np.ndarray:
        return calculator.get_feature_names(self.use_3D)

    def compute(self, mols: Sequence[Mol]) -> list[ArrayLike]:
        return [
            calculator.compute(
                mol=mol,
                use_3D=self.use_3D,
                timeout=self.timeout,
                max_ring_system_rank=self.max_ring_system_rank,
            )
            for mol in mols
        ]

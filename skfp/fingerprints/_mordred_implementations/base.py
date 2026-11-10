from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from rdkit.Chem import Mol

"""
Implementations of Mordred descriptors, selected with the ``implementation``
parameter of ``MordredFingerprint``.
"""


class BaseMordredImplementation(ABC):
    """
    Base class for Mordred descriptor implementations. Each implementation takes the
    ``MordredFingerprint`` parameters it uses.
    """

    @abstractmethod
    def get_feature_names(self) -> np.ndarray:
        """
        Names of the computed descriptors, in output order.
        """

    @abstractmethod
    def compute(self, mols: Sequence[Mol]) -> list[ArrayLike]:
        """
        Calculate descriptor values, array-like of shape ``(n_mols, n_features)``.
        Includes NaN where they are undefined.
        """

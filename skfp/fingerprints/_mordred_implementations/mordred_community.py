from collections.abc import Sequence

import numpy as np
from mordred import Calculator, descriptors
from numpy.typing import ArrayLike
from rdkit.Chem import Mol

from skfp.fingerprints._mordred_implementations.base import BaseMordredImplementation
from skfp.utils import get_conf_id

"""
The only module importing ``mordredcommunity``, an optional dependency.
``MordredFingerprint`` imports it only for ``implementation="mordredcommunity"``,
and explains how to install ``mordredcommunity`` if it is missing.
"""


class MordredCommunityImplementation(BaseMordredImplementation):
    """
    Mordred descriptors computed by the ``mordredcommunity`` library, with its own
    descriptor names and order.

    It has no ring system rank limit, applies ``timeout`` only to the detour matrix
    descriptors, and leaves 3D descriptors as NaN for flat conformers.
    """

    def __init__(self, use_3D: bool, timeout: float):
        self.use_3D = use_3D
        self._calculator = Calculator(
            descriptors, ignore_3D=not use_3D, config={"timeout": timeout}
        )

    def get_feature_names(self) -> np.ndarray:
        return np.asarray([str(d) for d in self._calculator.descriptors], dtype=object)

    def compute(self, mols: Sequence[Mol]) -> list[ArrayLike]:
        if not self.use_3D:
            return [self._calculator(mol) for mol in mols]

        # use the conformer conf_id points to, as the other 3D fingerprints do
        return [self._calculator(mol, id=get_conf_id(mol)) for mol in mols]

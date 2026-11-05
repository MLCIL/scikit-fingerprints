from collections.abc import Sequence

import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.implementations import MordredImplementation

"""
The only module importing ``mordredcommunity``, an optional dependency.
``MordredFingerprint`` imports it only for ``implementation="mordredcommunity"``,
and explains how to install ``mordredcommunity`` if it is missing.
"""


class MordredCommunityImplementation(MordredImplementation):
    """
    Mordred descriptors computed by the ``mordredcommunity`` library, with its own
    descriptor names and order.

    It has no ring system rank limit, so ``max_ring_system_rank`` is ignored, and it
    applies ``timeout`` only to the detour matrix descriptors. 3D descriptors always
    use the default conformer of a molecule, regardless of its ``conf_id`` property.
    """

    def __init__(self, use_3D: bool, timeout: float, max_ring_system_rank: int):
        super().__init__(use_3D, timeout, max_ring_system_rank)
        self._calculator = Calculator(
            descriptors, ignore_3D=not use_3D, config={"timeout": timeout}
        )

    def get_feature_names(self) -> np.ndarray:
        return np.asarray([str(d) for d in self._calculator.descriptors], dtype=object)

    def compute(self, mols: Sequence[Mol]) -> np.ndarray:
        X = [self._calculator(mol) for mol in mols]
        return np.array(X, dtype=np.float32)

from collections.abc import Sequence

import numpy as np
from mordred import Calculator, descriptors
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.implementations import MordredImplementation
from skfp.utils import get_conf_id

"""
The only module importing ``mordredcommunity``, an optional dependency.
``MordredFingerprint`` imports it only for ``implementation="mordredcommunity"``,
and explains how to install ``mordredcommunity`` if it is missing.
"""


class MordredCommunityImplementation(MordredImplementation):
    """
    Mordred descriptors computed by the ``mordredcommunity`` library, with its own
    descriptor names and order.

    It has no ring system rank limit, and applies ``timeout`` only to the detour
    matrix descriptors.
    """

    def __init__(self, use_3D: bool, timeout: float):
        self._calculator = Calculator(
            descriptors, ignore_3D=not use_3D, config={"timeout": timeout}
        )

    def get_feature_names(self) -> np.ndarray:
        return np.asarray([str(d) for d in self._calculator.descriptors], dtype=object)

    def compute(self, mols: Sequence[Mol]) -> np.ndarray:
        # get_conf_id needs a conformer; with -1, Mordred's default, molecules
        # without one get NaN 3D descriptors, as do the ones with a missing or a
        # flat conformer
        X = [
            self._calculator(mol, id=get_conf_id(mol) if mol.GetNumConformers() else -1)
            for mol in mols
        ]
        return np.array(X, dtype=np.float32)

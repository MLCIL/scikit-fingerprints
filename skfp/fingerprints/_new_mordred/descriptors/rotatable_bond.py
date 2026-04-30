"""
Rotatable bond descriptors implemented with RDKit bond counters.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np
from rdkit.Chem import Mol, rdMolDescriptors

FEATURE_NAMES = ["nRot", "RotRatio"]


def calc(mol_regular: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute the Mordred rotatable bond ratio descriptor.

    `RotRatio` is the number of rotatable bonds divided by the number of heavy
    atom bonds in the hydrogen-suppressed molecule.
    """
    n_bonds = mol_regular.GetNumBonds()
    n_rotatable = rdMolDescriptors.CalcNumRotatableBonds(mol_regular)
    rot_ratio = np.nan if n_bonds == 0 else n_rotatable / n_bonds

    return np.asarray([n_rotatable, rot_ratio], dtype=np.float32), FEATURE_NAMES

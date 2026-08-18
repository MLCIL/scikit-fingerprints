import numpy as np
from rdkit.Chem import Mol

"""
Vertex adjacency information descriptor.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["VAdjMat"]


def calc(mol_regular: Mol) -> np.ndarray:
    r"""
    Compute the Mordred vertex adjacency information descriptor.

    `VAdjMat` is defined as :math:`1 + \log_2(m)`, where :math:`m` is the number
    of heavy-heavy bonds. Returns NaN when :math:`m = 0`.
    """
    m = sum(
        1
        for bond in mol_regular.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() != 1
        and bond.GetEndAtom().GetAtomicNum() != 1
    )

    vadj_mat = np.nan if m == 0 else 1 + np.log2(m)

    return np.asarray([vadj_mat], dtype=np.float32)

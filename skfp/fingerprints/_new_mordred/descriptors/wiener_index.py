import numpy as np
from rdkit.Chem import Mol

from skfp.descriptors import polarity_number, wiener_index
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["WPath", "WPol"]


def calc(
    mol_regular: Mol, distance_matrix_regular: DistanceMatrix
) -> tuple[np.ndarray, list[str]]:
    values = np.array(
        [
            wiener_index(mol_regular, distance_matrix_regular.matrix),
            polarity_number(mol_regular, distance_matrix_regular.matrix),
        ],
        dtype=np.float32,
    )
    return values, FEATURE_NAMES

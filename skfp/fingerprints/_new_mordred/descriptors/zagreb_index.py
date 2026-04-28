import numpy as np
from rdkit.Chem import Mol

from skfp.descriptors import zagreb_index_m1, zagreb_index_m2
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Zagreb1", "Zagreb2", "mZagreb1", "mZagreb2"]


def calc(
    mol_regular: Mol, adjacency_matrix_regular: AdjacencyMatrix
) -> tuple[np.ndarray, list[str]]:
    values = np.array(
        [
            zagreb_index_m1(mol_regular, degree_vector=adjacency_matrix_regular.degree),
            zagreb_index_m2(mol_regular, degree_vector=adjacency_matrix_regular.degree),
            zagreb_index_m1(
                mol_regular,
                degree_vector=adjacency_matrix_regular.degree,
                modified=True,
            ),
            zagreb_index_m2(
                mol_regular,
                degree_vector=adjacency_matrix_regular.degree,
                modified=True,
            ),
        ],
        dtype=np.float32,
    )
    return values, FEATURE_NAMES

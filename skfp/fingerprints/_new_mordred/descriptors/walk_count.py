import numpy as np
from rdkit.Chem import Mol
from skfp.fingerprints._new_mordred.utils.graph_matrix import AdjacencyMatrix

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    # Molecular Walk Count
    "MWC01",
    "MWC02",
    "MWC03",
    "MWC04",
    "MWC05",
    "MWC06",
    "MWC07",
    "MWC08",
    "MWC09",
    "MWC10",
    # Total MWC
    "TMWC10",
    # Self-Returning Walk
    "SRW02",
    "SRW03",
    "SRW04",
    "SRW05",
    "SRW06",
    "SRW07",
    "SRW08",
    "SRW09",
    "SRW10",
    # Total SRW
    "TSRW10",
]

def calc(mol: Mol, adjacency_matrix: AdjacencyMatrix) -> tuple[np.ndarray, list[str]]:
    adjacency = adjacency_matrix.order().astype(np.float64)

    power = adjacency
    molecular_walk_counts: list[float] = []
    self_returning_walk_counts: list[float] = []

    for num_order in range(1, 11):
        if num_order > 1:
            power = adjacency_matrix.order(num_order).astype(np.float64)

        if num_order == 1:
            molecular_walk_counts.append(0.5 * float(power.sum()))
        else:
            molecular_walk_counts.append(np.log(float(power.sum()) + 1))
            self_returning_walk_counts.append(np.log(float(np.trace(power)) + 1))

    values = [
        *molecular_walk_counts,
        mol.GetNumAtoms() + sum(molecular_walk_counts),
        *self_returning_walk_counts,
        mol.GetNumAtoms() + sum(self_returning_walk_counts),
    ]

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES
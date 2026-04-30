"""
Carbon type descriptors implemented with direct RDKit atom access.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

from collections import defaultdict

import numpy as np
from rdkit.Chem import HybridizationType, Mol

FEATURE_NAMES = [
    "C1SP1",
    "C2SP1",
    "C1SP2",
    "C2SP2",
    "C3SP2",
    "C1SP3",
    "C2SP3",
    "C3SP3",
    "C4SP3",
    "HybRatio",
    "FCSP3",
]

_SP3_HYBRIDIZATIONS = {
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
}
_FEATURE_TO_COUNTS = {
    "C1SP1": (1, HybridizationType.SP),
    "C2SP1": (2, HybridizationType.SP),
    "C1SP2": (1, HybridizationType.SP2),
    "C2SP2": (2, HybridizationType.SP2),
    "C3SP2": (3, HybridizationType.SP2),
    "C1SP3": (1, HybridizationType.SP3),
    "C2SP3": (2, HybridizationType.SP3),
    "C3SP3": (3, HybridizationType.SP3),
    "C4SP3": (4, HybridizationType.SP3),
}


def calc(mol_kekulized: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred carbon type descriptors.
    """
    counts: defaultdict[tuple[int, HybridizationType], int] = defaultdict(int)
    hybridization_counts: defaultdict[HybridizationType, int] = defaultdict(int)
    num_carbons = 0

    for atom in mol_kekulized.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue

        num_carbons += 1
        hybridization = atom.GetHybridization()
        if hybridization in _SP3_HYBRIDIZATIONS:
            hybridization = HybridizationType.SP3

        carbon_neighbors = sum(
            neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
        )
        counts[(carbon_neighbors, hybridization)] += 1
        hybridization_counts[hybridization] += 1

    values: list[float] = []
    for name in FEATURE_NAMES[:-2]:
        carbon_neighbors, hybridization = _FEATURE_TO_COUNTS[name]
        values.append(counts[(carbon_neighbors, hybridization)])

    num_sp3 = hybridization_counts[HybridizationType.SP3]
    num_sp2 = hybridization_counts[HybridizationType.SP2]
    values.append(
        np.nan if num_sp2 == 0 and num_sp3 == 0 else num_sp3 / (num_sp2 + num_sp3)
    )
    values.append(0.0 if num_carbons == 0 else num_sp3 / num_carbons)

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES

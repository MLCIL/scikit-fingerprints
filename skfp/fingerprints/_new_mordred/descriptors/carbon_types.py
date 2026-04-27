"""Carbon type descriptors implemented with direct RDKit atom access.

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
]

_HYBRIDIZATION_TO_SP = {
    HybridizationType.SP: 1,
    HybridizationType.SP2: 2,
    HybridizationType.SP3: 3,
    HybridizationType.SP3D: 3,
    HybridizationType.SP3D2: 3,
}
_FEATURE_TO_COUNTS = {
    "C1SP1": (1, 1),
    "C2SP1": (2, 1),
    "C1SP2": (1, 2),
    "C2SP2": (2, 2),
    "C3SP2": (3, 2),
    "C1SP3": (1, 3),
    "C2SP3": (2, 3),
    "C3SP3": (3, 3),
    "C4SP3": (4, 3),
}


def calc(mol_kekulized: Mol) -> tuple[np.ndarray, list[str]]:
    """
    Compute Mordred carbon type descriptors in one pass over carbon atoms.

    The molecule must be kekulized to match Mordred `CarbonTypes` semantics.
    """
    counts: defaultdict[int | None, defaultdict[int, int]] = defaultdict(
        _default_int_dict
    )

    for atom in mol_kekulized.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue

        sp_type = _HYBRIDIZATION_TO_SP.get(atom.GetHybridization())
        carbon_neighbors = sum(
            neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
        )
        counts[sp_type][carbon_neighbors] += 1

    values: list[float] = []
    for name in FEATURE_NAMES[:-1]:
        carbon_neighbors, sp_type = _FEATURE_TO_COUNTS[name]
        values.append(counts[sp_type][carbon_neighbors])

    n_sp3 = sum(counts[3].values())
    n_sp2 = sum(counts[2].values())
    values.append(np.nan if n_sp2 == 0 and n_sp3 == 0 else n_sp3 / (n_sp2 + n_sp3))

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES


def _default_int_dict() -> defaultdict[int, int]:
    """
    Create a nested default dictionary for carbon type counts.
    """
    return defaultdict(int)

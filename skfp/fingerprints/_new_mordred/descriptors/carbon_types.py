from collections import defaultdict

import numpy as np
from rdkit.Chem import HybridizationType, Mol
from rdkit.Chem.rdMolDescriptors import CalcFractionCSP3

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_SP3_HYBRIDIZATIONS = {
    HybridizationType.SP3,
    HybridizationType.SP3D,
    HybridizationType.SP3D2,
}
_FEATURE_TO_COUNTS = {
    "sp carbons bonded to 1 carbon": (1, HybridizationType.SP),
    "sp carbons bonded to 2 carbons": (2, HybridizationType.SP),
    "sp2 carbons bonded to 1 carbon": (1, HybridizationType.SP2),
    "sp2 carbons bonded to 2 carbons": (2, HybridizationType.SP2),
    "sp2 carbons bonded to 3 carbons": (3, HybridizationType.SP2),
    "sp3 carbons bonded to 1 carbon": (1, HybridizationType.SP3),
    "sp3 carbons bonded to 2 carbons": (2, HybridizationType.SP3),
    "sp3 carbons bonded to 3 carbons": (3, HybridizationType.SP3),
    "sp3 carbons bonded to 4 carbons": (4, HybridizationType.SP3),
}

FEATURE_NAMES = [*_FEATURE_TO_COUNTS, "HybRatio", "Fsp3"]


def calc(mol_kekulized: Mol, mol_hydrogens: Mol) -> np.ndarray:
    """
    Count carbon atoms by carbon-neighbor degree and hybridization.

    ``Fsp3`` is RDKit's fraction of carbons with four neighbors, hydrogens
    included, on the hydrogen-explicit molecule, as in Mordred. That differs from
    the share of SP3-hybridized carbons for charged carbons, e.g. ``[CH2-]``.
    """
    counts = defaultdict(int)  # type: ignore[var-annotated]
    hybridization_counts = defaultdict(int)  # type: ignore[var-annotated]

    for atom in mol_kekulized.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue

        hybridization = atom.GetHybridization()
        if hybridization in _SP3_HYBRIDIZATIONS:
            hybridization = HybridizationType.SP3

        carbon_neighbors = sum(
            neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors()
        )
        counts[(carbon_neighbors, hybridization)] += 1
        hybridization_counts[hybridization] += 1

    # HybRatio uses aggregate hybridization counts, not neighbor counts.
    values = [counts[key] for key in _FEATURE_TO_COUNTS.values()]

    num_sp3 = hybridization_counts[HybridizationType.SP3]
    num_sp2 = hybridization_counts[HybridizationType.SP2]
    values.append(
        np.nan if num_sp2 == 0 and num_sp3 == 0 else num_sp3 / (num_sp2 + num_sp3)
    )
    values.append(CalcFractionCSP3(mol_hydrogens))

    return np.asarray(values, dtype=np.float32)

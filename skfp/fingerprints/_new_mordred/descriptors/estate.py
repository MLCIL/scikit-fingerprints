import itertools
from collections import defaultdict

import numpy as np
from rdkit.Chem import EState, Mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

# atom types defined in EState
_ATOM_TYPES = [
    "sLi",
    "ssBe",
    "ssssBe",
    "ssBH",
    "sssB",
    "ssssB",
    "sCH3",
    "dCH2",
    "ssCH2",
    "tCH",
    "dsCH",
    "aaCH",
    "sssCH",
    "ddC",
    "tsC",
    "dssC",
    "aasC",
    "aaaC",
    "ssssC",
    "sNH3",
    "sNH2",
    "ssNH2",
    "dNH",
    "ssNH",
    "aaNH",
    "tN",
    "sssNH",
    "dsN",
    "aaN",
    "sssN",
    "ddsN",
    "aasN",
    "ssssN",
    "sOH",
    "dO",
    "ssO",
    "aaO",
    "sF",
    "sSiH3",
    "ssSiH2",
    "sssSiH",
    "ssssSi",
    "sPH2",
    "ssPH",
    "sssP",
    "dsssP",
    "sssssP",
    "sSH",
    "dS",
    "ssS",
    "aaS",
    "dssS",
    "ddssS",
    "sCl",
    "sGeH3",
    "ssGeH2",
    "sssGeH",
    "ssssGe",
    "sAsH2",
    "ssAsH",
    "sssAs",
    "sssdAs",
    "sssssAs",
    "sSeH",
    "dSe",
    "ssSe",
    "aaSe",
    "dssSe",
    "ddssSe",
    "sBr",
    "sSnH3",
    "ssSnH2",
    "sssSnH",
    "ssssSn",
    "sI",
    "sPbH3",
    "ssPbH2",
    "sssPbH",
    "ssssPb",
]

FEATURE_NAMES = list(
    itertools.chain.from_iterable(
        (
            f"N{atom_type}",  # count
            f"S{atom_type}",  # sum
            f"MAX{atom_type}",  # max
            f"MIN{atom_type}",  # min
        )
        for atom_type in _ATOM_TYPES
    )
)


def calc(mol: Mol) -> np.ndarray:
    """
    EState descriptors.

    Quantifies electronic and topological information based on predefined
    atom types.
    """
    atom_types = EState.TypeAtoms(mol)
    indices = EState.EStateIndices(mol)

    # EState index values of all atoms carrying each atom type
    matched = defaultdict(list)
    for labels, index in zip(atom_types, indices, strict=True):
        for label in labels:
            matched[label].append(index)

    values = []
    for atom_type in _ATOM_TYPES:
        type_indices = matched.get(atom_type, [])

        count = len(type_indices)
        total = float(sum(type_indices))
        if type_indices:
            maximum = max(type_indices)
            minimum = min(type_indices)
        else:
            maximum = np.nan
            minimum = np.nan

        values.extend([count, total, maximum, minimum])

    return np.asarray(values, dtype=np.float32)

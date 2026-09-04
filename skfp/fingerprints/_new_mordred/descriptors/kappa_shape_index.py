import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
Kappa shape index descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["Kier1", "Kier2", "Kier3"]


def calc(props: AtomicProperties, subgraphs: Subgraphs) -> np.ndarray:
    """
    Kappa shape indices of orders 1 to 3.

    Each order compares how many paths of that many bonds the molecule has
    against the counts of the least and most path-rich graphs on the same number
    of atoms, so a molecule scores higher the closer its shape is to a chain.

    NaN is returned for an order the molecule spans no paths of.

    Based on Kier, L. B. (1985). A shape index from molecular graphs.
    Quantitative Structure-Activity Relationships, 4(3), 109-116.
    https://doi.org/10.1002/qsar.19850040303
    """
    num_atoms = props.num_atoms
    numerators = [
        num_atoms * (num_atoms - 1) ** 2,
        (num_atoms - 1) * (num_atoms - 2) ** 2,
        (num_atoms - 2) ** 2 * (num_atoms - 3)
        if num_atoms % 2 == 0
        else (num_atoms - 1) * (num_atoms - 3) ** 2,
    ]

    values = [
        numerator / (num_paths * num_paths) if num_paths else float("nan")
        for numerator, num_paths in zip(
            numerators,
            (len(subgraphs.paths(order).bond_idxs) for order in (1, 2, 3)),
            strict=True,
        )
    ]
    return np.asarray(values, dtype=np.float32)

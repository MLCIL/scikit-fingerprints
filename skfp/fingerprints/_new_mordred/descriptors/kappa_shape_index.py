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

    # the least path-rich graph is the linear chain, for every order
    min_paths = [num_atoms - order for order in (1, 2, 3)]

    # leaves per hub, split as evenly as the atom count allows
    if num_atoms % 2 == 0:
        max_paths_3 = ((num_atoms - 2) / 2) ** 2
    else:
        max_paths_3 = (num_atoms - 1) / 2 * ((num_atoms - 3) / 2)

    max_paths = [
        num_atoms * (num_atoms - 1) / 2,  # complete graph, every atom pair bonded
        (num_atoms - 1) * (num_atoms - 2) / 2,  # star, every leaf pair via the hub
        max_paths_3,  # two stars joined by a bond, leaves split evenly
    ]

    # orders 1 and 2 are normalized by 2, order 3 by 4
    scales = [2, 2, 4]

    num_paths = [len(subgraphs.paths(order).bond_idxs) for order in (1, 2, 3)]

    values = [
        scale * num_max_paths * num_min_paths / (num_mol_paths * num_mol_paths)
        if num_mol_paths
        else float("nan")
        for scale, num_max_paths, num_min_paths, num_mol_paths in zip(
            scales, max_paths, min_paths, num_paths, strict=True
        )
    ]
    return np.asarray(values, dtype=np.float32)

import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

MAX_ORDER = 10

FEATURE_NAMES = [
    # molecular path count, orders 2-10
    "MPC2",
    "MPC3",
    "MPC4",
    "MPC5",
    "MPC6",
    "MPC7",
    "MPC8",
    "MPC9",
    "MPC10",
    # total MPC (orders 0-10)
    "TMPC10",
    # pi-weighted path count (log scale), orders 1-10
    "piPC1",
    "piPC2",
    "piPC3",
    "piPC4",
    "piPC5",
    "piPC6",
    "piPC7",
    "piPC8",
    "piPC9",
    "piPC10",
    # total pi-weighted path count (log scale), orders 0-10
    "TpiPC10",
]


def calc(props: AtomicProperties, subgraphs: Subgraphs) -> np.ndarray:
    """
    Path count descriptors.

    path_counts[k]: number of self-avoiding paths with exactly k bonds
    pi_counts[k]: sum over those paths of the product of their bond orders
    """
    # up to MAX_ORDER (inclusive) atoms
    path_counts = [0.0] * (MAX_ORDER + 1)
    pi_counts = [0.0] * (MAX_ORDER + 1)

    # 0th order is a single atom
    n_atoms = props.num_atoms
    path_counts[0] = n_atoms
    pi_counts[0] = n_atoms

    bond_orders = props.bond_orders

    for order in range(1, MAX_ORDER + 1):
        bond_idxs = subgraphs.path_bond_idxs(order)
        path_counts[order] = float(len(bond_idxs))
        pi_counts[order] = float(bond_orders[bond_idxs].prod(axis=1).sum())

    total_path_count = sum(path_counts)

    log_pi_counts = [np.log(pi_counts[order] + 1) for order in range(1, MAX_ORDER + 1)]
    total_log_pi_count = np.log(sum(pi_counts) + 1)

    values = [
        *path_counts[2:],
        total_path_count,
        *log_pi_counts,
        total_log_pi_count,
    ]

    return np.asarray(values, dtype=np.float32)

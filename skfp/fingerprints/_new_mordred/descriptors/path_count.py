from itertools import chain

import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import (
    MAX_ORDER as SUBGRAPH_MAX_ORDER,
)
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


def calc(
    mol: Mol, props: AtomicProperties, subgraphs: Subgraphs
) -> tuple[np.ndarray, list[str]]:
    """
    Path count descriptors.

    path_counts[k]: number of self-avoiding paths with exactly k bonds
    pi_counts[k]: sum over those paths of the product of their bond orders

    A self-avoiding path is exactly a connected subgraph of the "path" type, so up
    to :data:`~skfp.fingerprints._new_mordred.utils.subgraphs.MAX_ORDER` the paths
    come from the shared subgraph enumeration; the longer ones are enumerated here.
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
        if order <= SUBGRAPH_MAX_ORDER:
            bond_idxs = subgraphs.path_bond_idxs(order)
        else:
            bond_idxs = _self_avoiding_path_bond_idxs(mol, props, order)

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

    return np.asarray(values, dtype=np.float32), FEATURE_NAMES


def _self_avoiding_path_bond_idxs(
    mol: Mol, props: AtomicProperties, order: int
) -> np.ndarray:
    """
    Bond indices of every self-avoiding path with ``order`` bonds, shape
    ``(n_paths, order)``.
    """
    paths = Chem.FindAllPathsOfLengthN(mol, length=order, useBonds=True)
    num_paths = len(paths)
    if num_paths == 0:
        return np.empty((0, order), dtype=np.intp)

    # each inner vector is copied with [:] before being read: measurably faster
    # than iterating it in place through RDKit's Python-level vector iterator,
    # which dominates the cost of this descriptor family
    bond_idxs = np.fromiter(
        chain.from_iterable(path[:] for path in paths),
        dtype=np.intp,
        count=num_paths * order,
    ).reshape(num_paths, order)

    # RDKit may return self-returning paths (e.g. around small rings); path counts
    # only consider self-avoiding ones, which span exactly order + 1 distinct atoms
    bond_atoms = np.stack([props.bond_begin_idxs, props.bond_end_idxs], axis=1)
    path_atoms = np.sort(bond_atoms[bond_idxs].reshape(num_paths, 2 * order), axis=1)
    n_distinct_atoms = np.count_nonzero(np.diff(path_atoms, axis=1), axis=1) + 1

    return bond_idxs[n_distinct_atoms == order + 1]

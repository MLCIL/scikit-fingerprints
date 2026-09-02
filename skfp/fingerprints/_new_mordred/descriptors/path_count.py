import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.subgraphs import (
    SUBGRAPH_MAX_NUM_BONDS,
    Paths,
    Subgraphs,
)

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
    # order 0 is a lone atom: one path per atom, with no bond orders to multiply
    path_counts = [float(props.num_atoms)]
    pi_counts = [float(props.num_atoms)]

    for order in range(1, MAX_ORDER + 1):
        # enumerated subgraphs (including paths) shared with Chi descriptors only
        # reach SUBGRAPH_MAX_NUM_BONDS, past that we need to compute paths separately
        if order <= SUBGRAPH_MAX_NUM_BONDS:
            paths = subgraphs.paths(order)
        else:
            paths = _grow_paths(paths, subgraphs)

        bond_idxs = paths.bond_idxs
        path_counts.append(float(len(bond_idxs)))
        pi_counts.append(float(props.bond_orders[bond_idxs].prod(axis=1).sum()))

    # piPC starts at 1
    pi_log_counts = [np.log(pi_count + 1) for pi_count in pi_counts[1:]]

    values = [
        *path_counts[2:],  # MPC starts at order 2
        sum(path_counts),
        *pi_log_counts,
        np.log(sum(pi_counts) + 1),
    ]

    return np.asarray(values, dtype=np.float32)


def _grow_paths(paths: Paths, subgraphs: Subgraphs) -> Paths:
    """
    Extend every path by one bond, at either of its two ends.
    """
    halves = [_grow_at_end(paths, subgraphs, end) for end in (0, 1)]

    bond_idxs = np.concatenate([half.bond_idxs for half in halves])
    bond_idxs.sort(axis=1)  # a path is a set of bonds, so put the rows in order
    return Paths(
        bond_idxs,
        np.concatenate([half.atom_idxs for half in halves]),
        np.concatenate([half.end_atoms for half in halves]),
    )


def _grow_at_end(paths: Paths, subgraphs: Subgraphs, end: int) -> Paths:
    """
    Extend every path by one bond at one nominated end, ``end`` being 0 or 1.

    Some paths yield several extensions and some none, so the result has its own
    number of rows and its own ordering.
    """
    bond_idxs, atom_idxs = paths.bond_idxs, paths.atom_idxs
    end_atoms = paths.end_atoms
    grown_atoms = end_atoms[:, end]

    # calculate bonds from a given atom as candidates to grow a path
    incident = subgraphs.bonds_of_atom[grown_atoms]
    is_bond = incident >= 0  # the rest is padding

    # flatten rows into one candidate bond per entry
    # path_of[i] is the path that candidates[i] would extend
    path_of, slot_of = np.nonzero(is_bond)
    candidates = incident[path_of, slot_of]

    # a bond holds both of its atoms, so subtracting the one being grown from
    # leaves the atom on the far side: the one this extension would add
    new_atoms = subgraphs.bond_atoms[candidates].sum(axis=1) - grown_atoms[path_of]

    # reject already visited atoms and bonds
    atoms_on_path = atom_idxs[path_of]  # (n_candidates, n_atoms_per_path)
    self_avoiding = (atoms_on_path != new_atoms[:, None]).all(axis=1)

    # a grown path ends at the atom just added, and at the end left alone
    # either of those two could have been the one added last, which is ambiguous
    # requiring the added atom to have lower number removes potential duplicates
    other_end = end_atoms[path_of, 1 - end]
    keep = self_avoiding & (new_atoms < other_end)

    candidates, path_of = candidates[keep], path_of[keep]
    new_atoms, other_end = new_atoms[keep], other_end[keep]

    return Paths(
        # each survivor inherits its parent's bonds and atoms, plus the ones added
        np.concatenate([bond_idxs[path_of], candidates[:, None]], axis=1),
        np.concatenate([atom_idxs[path_of], new_atoms[:, None]], axis=1),
        # the grown end moved to the atom just added; the other end stayed put
        np.stack([new_atoms, other_end], axis=1),
    )

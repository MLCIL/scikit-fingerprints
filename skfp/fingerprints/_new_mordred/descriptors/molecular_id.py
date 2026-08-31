from math import sqrt

import numpy as np

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.periodic_table import HALOGEN_ATOMIC_NUMS

"""
Molecular ID descriptors.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = [
    "MID",
    "AMID",
    "MID_h",
    "AMID_h",
    "MID_C",
    "AMID_C",
    "MID_N",
    "AMID_N",
    "MID_O",
    "AMID_O",
    "MID_X",
    "AMID_X",
]

# a path stops being extended once the product of its edge weights reaches this
# limit, as anything longer would contribute less than Mordred's 1e-10 epsilon;
# Mordred spells the same number as int(1 / eps ** 2)
_WEIGHT_PRODUCT_LIMIT = 10**20

_HYDROGEN_ATOMIC_NUM = 1
_CARBON_ATOMIC_NUM = 6
_NITROGEN_ATOMIC_NUM = 7
_OXYGEN_ATOMIC_NUM = 8
_HALOGEN_ATOMIC_NUMS = np.asarray(sorted(HALOGEN_ATOMIC_NUMS), dtype=np.intp)


def calc(props_regular: AtomicProperties, n_frags: int) -> np.ndarray:
    """
    Molecular ID descriptors, sums of atomic IDs over a selected set of atoms.

    The features come in pairs: a sum over the selected atoms and that same sum
    averaged over the molecule. Selections are all atoms (``MID``), heteroatoms
    (``MID_h``), carbons, nitrogens, oxygens and halogens (``MID_X``).

    Atomic IDs are undefined for disconnected molecules, so NaN is returned for
    those.
    """
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    atom_ids = _atom_ids(props_regular)
    atomic_nums = props_regular.atomic_nums
    # a heteroatom here is anything but carbon and hydrogen, unlike the
    # carbon-only definition AtomicProperties.is_hetero uses
    is_hetero = (atomic_nums != _HYDROGEN_ATOMIC_NUM) & (
        atomic_nums != _CARBON_ATOMIC_NUM
    )
    selections = [
        np.ones(props_regular.num_atoms, dtype=bool),
        is_hetero,
        atomic_nums == _CARBON_ATOMIC_NUM,
        atomic_nums == _NITROGEN_ATOMIC_NUM,
        atomic_nums == _OXYGEN_ATOMIC_NUM,
        np.isin(atomic_nums, _HALOGEN_ATOMIC_NUMS),
    ]

    values = []
    for selection in selections:
        molecular_id = atom_ids[selection].sum()
        # averaged over every atom of the molecule, not over the selected ones
        values += [molecular_id, molecular_id / props_regular.num_atoms]

    return np.asarray(values, dtype=np.float32)


def _atom_ids(props: AtomicProperties) -> np.ndarray:
    """
    Atomic ID of every atom: one plus half the sum, over all simple paths
    starting at that atom, of the inverse square root of the product of the
    path's edge weights.
    """
    adjacency = _weighted_adjacency(props)
    num_atoms = props.num_atoms
    visited = bytearray(num_atoms)

    return np.fromiter(
        (
            1.0 + _sum_over_paths(adjacency, visited, start, 1) / 2.0
            for start in range(num_atoms)
        ),
        dtype=np.float64,
        count=num_atoms,
    )


def _weighted_adjacency(props: AtomicProperties) -> list[list[tuple[int, int]]]:
    """
    Neighbors of every atom paired with their bond weight, the product of the
    degrees of the two bonded atoms.

    Kept as nested Python lists of plain ints rather than NumPy arrays: the path
    search below reads them one edge at a time, where boxing NumPy scalars costs
    far more than the lookup itself.
    """
    degrees = props.degrees
    begin_idxs = props.bond_begin_idxs.tolist()
    end_idxs = props.bond_end_idxs.tolist()
    weights = (degrees[props.bond_begin_idxs] * degrees[props.bond_end_idxs]).tolist()

    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(props.num_atoms)]
    for begin, end, weight in zip(begin_idxs, end_idxs, weights, strict=True):
        adjacency[begin].append((end, weight))
        adjacency[end].append((begin, weight))

    return adjacency


def _sum_over_paths(
    adjacency: list[list[tuple[int, int]]],
    visited: bytearray,
    atom: int,
    weight_product: int,
) -> float:
    """
    Sum of the inverse square root of the edge weight product over every simple
    path that extends the one ending at ``atom``.

    ``visited`` marks the atoms already on the path and is restored on the way
    out, so the same buffer serves every starting atom.
    """
    visited[atom] = 1
    total = 0.0

    for neighbor, weight in adjacency[atom]:
        if visited[neighbor]:
            continue

        product = weight_product * weight
        total += 1.0 / sqrt(product)
        # a longer path would only add terms below the epsilon, so stop here
        if product < _WEIGHT_PRODUCT_LIMIT:
            total += _sum_over_paths(adjacency, visited, neighbor, product)

    visited[atom] = 0
    return total

from collections import Counter

import numpy as np
from rdkit.Chem import Atom, Mol
from rdkit.Chem.rdchem import BondType

from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    BOND_ORDERS,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import atoms_apply_func

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_VARIANTS = (
    "information_content",
    "total_information_content",
    "structural_information_content",
    "bonding_information_content",
    "complementary_information_content",
    "modified_information_content",
    "Z_modified_information_content",
)
_MAX_ORDER = 5

FEATURE_NAMES = [
    f"{variant}_order_{order}"
    for variant in _VARIANTS
    for order in range(_MAX_ORDER + 1)
]


def calc(
    mol_hydrogens: Mol,
    atomic_props_hydrogens: AtomicProperties,
    kekulized_bond_types: np.ndarray,
) -> np.ndarray:
    """
    Information content descriptors, of orders 0 to ``_MAX_ORDER``.

    Atoms are grouped by the neighborhood of a given order around them, into classes
    of atoms coded alike. One class holds n of the molecule's A atoms, a share
    p = n / A of them; B is the sum of the bond orders over all bonds; and every sum
    below runs over the classes, one term each:

    - neighborhood information content, the entropy of those shares:
        IC = sum of -p * log2(p)
    - neighborhood total information content, that entropy over the whole molecule
      rather than per atom:
        TIC = A * IC
    - structural information content, IC against the log2(A) that A atoms can reach
      at most; NaN for a single atom:
        SIC = IC / log2(A)
    - bonding information content, the same against the bonds instead; NaN when
      B <= 1:
        BIC = IC / log2(B)
    - complementary information content, how far IC falls short of that maximum:
        CIC = log2(A) - IC
    - modified information content index, IC with every term weighted by the mass of
      the class's atoms:
        MIC = sum of -mass * p * log2(p)
    - Z-modified information content index, weighted instead by their atomic number
      and by the size of the class:
        ZMIC = sum of -n * Z * p * log2(p)

    The molecule carries explicit hydrogens, and its aromatic bonds are kekulized.
    """
    num_atoms = atomic_props_hydrogens.num_atoms  # A
    if num_atoms == 0:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    atomic_nums = atomic_props_hydrogens.atomic_nums  # Z
    masses = atoms_apply_func(Atom.GetMass, mol_hydrogens, np.float64)

    # the kekulized types cover the bonds between heavy atoms only, as they were read
    # off the molecule before hydrogens were added; adding those appends their bonds
    # last, and every one of them is single
    n_h_bonds = atomic_props_hydrogens.num_bonds - len(kekulized_bond_types)
    bond_types = np.concatenate(
        [kekulized_bond_types, np.full(n_h_bonds, int(BondType.SINGLE), np.intp)]
    )
    bond_order_sum = BOND_ORDERS[bond_types].sum()  # B

    codes = _neighborhood_codes(atomic_props_hydrogens, bond_types)
    class_size_per_atom = _class_size_per_atom(codes)

    ic = _shannon_entropy(class_size_per_atom)
    tic = num_atoms * ic
    sic = ic / (np.log2(num_atoms) if num_atoms > 1 else np.nan)
    bic = ic / (np.log2(bond_order_sum) if bond_order_sum > 1 else np.nan)
    cic = np.log2(num_atoms) - ic
    mic = _shannon_entropy(class_size_per_atom, masses)
    zmic = _shannon_entropy(class_size_per_atom, atomic_nums * class_size_per_atom)

    return np.concatenate((ic, tic, sic, bic, cic, mic, zmic), dtype=np.float32)


def _neighborhood_codes(
    props: AtomicProperties, bond_types: np.ndarray
) -> list[list[int | tuple]]:
    """
    Code of the neighborhood of every order around every atom, as ``_MAX_ORDER + 1``
    lists holding one code per atom.

    A neighborhood is grown around each atom, one bond further per order, and coded
    as the sorted paths running from the atom to the leaves, spelling out the element,
    the degree and the bond type at every step. Atoms coded alike share a class.

    A code of order 1 or higher is a tuple holding those paths, sorted, the ones that
    repeat among them kept. Order 0 reaches no bond at all and codes an atom by its
    atomic number alone, so its codes are plain integers.
    """
    neighbors = _neighbors(props, bond_types)

    # the element and the degree of every atom, which is all a path spells out about
    # the atoms it passes through; kept as a list of tuples, as the growing loop below
    # reads it atom by atom rather than in bulk
    atom_keys = list(
        zip(props.atomic_nums.tolist(), props.degrees.tolist(), strict=True)
    )

    # order 0 reaches no bond at all, so an atom's element is its whole code
    codes: list[list[int | tuple]] = [props.atomic_nums.tolist()]
    codes += [[] for _ in range(_MAX_ORDER)]

    for root in range(props.num_atoms):
        visited = {root}
        # the atoms the current order reaches, each with the path that leads to it
        # from the root. A path holds an atom key at every even position and the type
        # of the bond crossed to the next atom at every odd one, and starts out
        # holding the key of the root alone
        frontier: list[tuple[int, tuple]] = [(root, (atom_keys[root],))]
        # paths that ran out of unvisited atoms to grow into before the current
        # order; they end at a leaf of the neighborhood, so they keep taking part in
        # the code of every order from then on
        finished_paths: list[tuple] = []

        for order in range(1, _MAX_ORDER + 1):
            next_frontier: list[tuple[int, tuple]] = []
            for atom_idx, path in frontier:
                # marking the atom visited here, rather than when it entered the
                # frontier, lets a path step sideways into another atom of the same
                # order, which is what happens in a ring
                visited.add(atom_idx)
                # one order of growth appends the type of the bond crossed and the
                # key of the atom it reaches, keeping the path alternating
                grown = [
                    (nbr, (*path, bond_type, atom_keys[nbr]))
                    for nbr, bond_type in neighbors[atom_idx]
                    if nbr not in visited
                ]
                if grown:
                    next_frontier.extend(grown)
                else:
                    finished_paths.append(path)

            # the neighborhood is made of the paths that reach its leaves, sorted so
            # that two atoms grown alike spell out one and the same code
            codes[order].append(
                tuple(sorted(finished_paths + [path for _, path in next_frontier]))
            )
            frontier = next_frontier

    return codes


def _class_size_per_atom(codes: list[list[int | tuple]]) -> np.ndarray:
    """
    Size of the class every atom falls into, shaped ``(_MAX_ORDER + 1, num_atoms)``.

    Atoms carrying the same neighborhood code of a given order make up one class of
    that order. Every entry is the size of the class holding that atom, repeated once
    per atom of the class rather than kept once per class.
    """
    class_size_per_atom = np.empty((len(codes), len(codes[0])), dtype=np.intp)
    for order, order_codes in enumerate(codes):
        # an order holds one code per atom, so counting how often a whole code occurs
        # among them counts the atoms of the class it stands for
        class_counts = Counter(order_codes)
        class_size_per_atom[order] = [class_counts[code] for code in order_codes]
    return class_size_per_atom


def _neighbors(
    props: AtomicProperties, bond_types: np.ndarray
) -> list[list[tuple[int, int]]]:
    """
    Neighbors of every atom, each with the type of the bond leading to it.
    """
    neighbors: list[list[tuple[int, int]]] = [[] for _ in range(props.num_atoms)]
    for begin, end, bond_type in zip(
        props.bond_begin_idxs.tolist(),
        props.bond_end_idxs.tolist(),
        bond_types.tolist(),
        strict=True,
    ):
        neighbors[begin].append((end, bond_type))
        neighbors[end].append((begin, bond_type))
    return neighbors


def _shannon_entropy(
    class_size_per_atom: np.ndarray, weights: np.ndarray | float = 1.0
) -> np.ndarray:
    """
    Entropy of the class sizes of every order, each atom weighted as given.

    ``class_size_per_atom`` holds the size of the class of every atom rather than the
    size of every class, so the sum runs over atoms: a class of n atoms contributes
    its term n times, which is the same as weighting that term by n.
    """
    num_atoms = class_size_per_atom.shape[1]
    probabilities = class_size_per_atom / num_atoms
    return -(weights * np.log2(probabilities)).sum(axis=1) / num_atoms

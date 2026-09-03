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

    n_h_bonds = atomic_props_hydrogens.num_bonds - len(kekulized_bond_types)
    bond_types = np.concatenate(
        [kekulized_bond_types, np.full(n_h_bonds, int(BondType.SINGLE), np.intp)]
    )
    bond_order_sum = BOND_ORDERS[bond_types].sum()  # B

    class_sizes = _class_sizes(atomic_props_hydrogens, bond_types)

    ic = _shannon_entropy(class_sizes)
    tic = num_atoms * ic
    sic = ic / (np.log2(num_atoms) if num_atoms > 1 else np.nan)
    bic = ic / (np.log2(bond_order_sum) if bond_order_sum > 1 else np.nan)
    cic = np.log2(num_atoms) - ic
    mic = _shannon_entropy(class_sizes, masses)
    zmic = _shannon_entropy(class_sizes, atomic_nums * class_sizes)

    return np.concatenate((ic, tic, sic, bic, cic, mic, zmic), dtype=np.float32)


def _class_sizes(props: AtomicProperties, bond_types: np.ndarray) -> np.ndarray:
    """
    Size of the class every atom falls into, shaped ``(_MAX_ORDER + 1, num_atoms)``.

    A neighborhood is grown around each atom, one bond further per order, and coded
    as the sorted paths running from the atom to the leaves, spelling out the element,
    the degree and the bond type at every step. Atoms coded alike share a class.
    Order 0 codes an atom by its element alone.

    Every entry is the size of the class holding that atom, repeated once per atom of
    the class rather than kept once per class, which is what lets the entropy be summed
    over atoms.
    """
    neighbors = _neighbors(props, bond_types)
    atomic_nums = props.atomic_nums.tolist()
    degrees = props.degrees.tolist()

    interned: dict[tuple, int] = {}

    def intern(path: tuple) -> int:
        return interned.setdefault(path, len(interned))

    codes = np.empty((_MAX_ORDER + 1, props.num_atoms), dtype=np.intp)
    for root in range(props.num_atoms):
        codes[0, root] = atomic_nums[root]
        visited = {root}
        leaves = [(root, intern((atomic_nums[root], degrees[root])))]
        dead_ends: list[int] = []

        for order in range(1, _MAX_ORDER + 1):
            grown: list[tuple[int, int]] = []
            for atom_idx, path in leaves:
                visited.add(atom_idx)
                children = [
                    (nbr, intern((path, bond_type, atomic_nums[nbr], degrees[nbr])))
                    for nbr, bond_type in neighbors[atom_idx]
                    if nbr not in visited
                ]
                if children:
                    grown.extend(children)
                else:
                    dead_ends.append(path)

            codes[order, root] = intern(
                tuple(sorted(dead_ends + [path for _, path in grown]))
            )
            leaves = grown

    class_sizes = np.empty_like(codes)
    for order, order_codes in enumerate(codes):
        _, class_idxs, counts = np.unique(
            order_codes, return_inverse=True, return_counts=True
        )
        class_sizes[order] = counts[class_idxs]
    return class_sizes


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
    class_sizes: np.ndarray, weights: np.ndarray | float = 1.0
) -> np.ndarray:
    """
    Entropy of the class sizes of every order, each atom weighted as given.

    ``class_sizes`` holds the size of the class of every atom rather than the size of
    every class, so the entropy is summed over atoms: a class of n atoms contributes
    its term n times, which is what weighting that term by n amounts to.
    """
    num_atoms = class_sizes.shape[1]
    probabilities = class_sizes / num_atoms
    return -(weights * np.log2(probabilities)).sum(axis=1) / num_atoms

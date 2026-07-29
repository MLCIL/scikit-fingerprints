"""
Mordred descriptors implemented as direct RDKit wrappers.

This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

import numpy as np
from rdkit.Chem import Crippen, Descriptors, Mol, rdMolDescriptors
from rdkit.Chem.EState import EState_VSA

from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.descriptor_evaluation import safe_value
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.ragged import ragged_indices, run_starts

FEATURE_NAMES_2D = [
    "BalabanJ",
    "BertzCT",
    "nHBAcc",
    "nHBDon",
    "LabuteASA",
    *[f"PEOE_VSA{i}" for i in range(1, 14)],
    *[f"SMR_VSA{i}" for i in range(1, 10)],
    *[f"SlogP_VSA{i}" for i in range(1, 12)],
    *[f"EState_VSA{i}" for i in range(1, 11)],
    *[f"VSA_EState{i}" for i in range(1, 10)],
    "SLogP",
    "SMR",
    "TopoPSA(NO)",
    "TopoPSA",
    "MW",
    "AMW",
]

FEATURE_NAMES_3D = ["MOMI-Z", "MOMI-Y", "MOMI-X", "PBF"]


def _calc_moe_type_descriptors(mol: Mol) -> list[float]:
    """
    Compute RDKit MOE-type VSA descriptors.

    Each VSA group splits approximate molecular surface area into bins based on
    atom-level properties such as partial charge, molar refractivity, logP, and
    E-State values.

    The charge, refractivity and logP groups are read from RDKit's C++ functions,
    which return all bins of a group at once; the per-bin functions in
    ``rdkit.Chem.MolSurf`` are Python wrappers around the very same values.
    """
    return [
        *rdMolDescriptors.PEOE_VSA_(mol)[:13],
        *rdMolDescriptors.SMR_VSA_(mol)[:9],
        *rdMolDescriptors.SlogP_VSA_(mol)[:11],
        *[getattr(EState_VSA, f"EState_VSA{idx}")(mol) for idx in range(1, 11)],
        *[getattr(EState_VSA, f"VSA_EState{idx}")(mol) for idx in range(1, 10)],
    ]


def _balaban_j(props: AtomicProperties, distances: np.ndarray) -> float:
    """
    Balaban's J index, an average distance-based connectivity index.

    See Balaban, Chem. Phys. Lett. 89, 399-404 (1982). Computes the same value as
    RDKit's ``GraphDescriptors.BalabanJ``, which sums over the atom pairs in Python.

    Mordred, and therefore this implementation, feeds it the plain topological
    distances rather than the bond-order-weighted ones of Balaban's paper.
    """
    num_bonds = props.num_bonds
    cyclomatic_number = num_bonds - props.num_atoms + 1
    if cyclomatic_number + 1 == 0:
        return 0.0

    distance_sums = distances.sum(axis=1)
    bond_terms = 1.0 / np.sqrt(
        distance_sums[props.bond_begin_idxs] * distance_sums[props.bond_end_idxs]
    )
    return num_bonds / (cyclomatic_number + 1) * bond_terms.sum()


def _symmetry_classes(weighted_distances: np.ndarray) -> np.ndarray:
    """
    Group atoms into topologically equivalent classes, two atoms being equivalent
    when their sorted vectors of distances to all other atoms agree.

    RDKit compares these vectors as strings of four decimals, hence the rounding.
    """
    distance_vectors = np.round(np.sort(weighted_distances, axis=1), 4)
    return np.unique(distance_vectors, axis=0, return_inverse=True)[1]


def _bertz_connection_counts(
    props: AtomicProperties, symmetry_classes: np.ndarray, bond_orders: np.ndarray
) -> np.ndarray:
    """
    Count the "connections" of every distinct kind of connection.

    A connection is a pair of bonds meeting at an atom, counted with the product of
    their bond orders, or a pair of the parallel bonds that a multiple bond stands
    for. Two connections are of the same kind when they involve the same symmetry
    classes, so counting them by kind is a grouping over class pairs and triples.
    """
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs

    # a bond of order n stands for n parallel bonds, which pair up among themselves
    multiple = bond_orders > 1
    multiple_orders = bond_orders[multiple]
    within_bond_counts = multiple_orders * (multiple_orders - 1) / 2
    within_bond_kinds = np.sort(
        np.stack(
            [symmetry_classes[begins[multiple]], symmetry_classes[ends[multiple]]]
        ),
        axis=0,
    )

    # bonds listed once per end, grouped by that end: the atom pairs meet there
    hinges = np.concatenate([begins, ends])
    neighbors = np.concatenate([ends, begins])
    orders = np.concatenate([bond_orders, bond_orders])
    by_hinge = np.argsort(hinges, kind="stable")
    neighbors, orders = neighbors[by_hinge], orders[by_hinge]

    num_bonds_at_atom = np.bincount(hinges, minlength=props.num_atoms)
    first_bond_at_atom = run_starts(num_bonds_at_atom)
    # all ordered pairs of the bonds at an atom, of which the halves below the
    # diagonal are the unordered ones
    hinge_of_pair, within = ragged_indices(num_bonds_at_atom * num_bonds_at_atom)
    num_bonds_at_hinge = num_bonds_at_atom[hinge_of_pair]
    first = first_bond_at_atom[hinge_of_pair] + within // num_bonds_at_hinge
    second = first_bond_at_atom[hinge_of_pair] + within % num_bonds_at_hinge
    pairs = first < second
    hinge_of_pair, first, second = hinge_of_pair[pairs], first[pairs], second[pairs]

    at_hinge_counts = orders[first] * orders[second]
    neighbor_classes = np.sort(
        np.stack(
            [symmetry_classes[neighbors[first]], symmetry_classes[neighbors[second]]]
        ),
        axis=0,
    )
    at_hinge_kinds = np.stack(
        [neighbor_classes[0], symmetry_classes[hinge_of_pair], neighbor_classes[1]]
    )

    # the two flavors of connection are of different kinds by construction, so a
    # kind is identified by the classes it lists, padded to the same width
    kinds = np.concatenate(
        [
            np.pad(within_bond_kinds, ((0, 1), (0, 0)), constant_values=-1),
            at_hinge_kinds,
        ],
        axis=1,
    )
    counts = np.concatenate([within_bond_counts, at_hinge_counts])
    kind_idxs = np.unique(kinds, axis=1, return_inverse=True)[1]
    return np.bincount(kind_idxs, weights=counts)


def _bertz_complexity(props: AtomicProperties, weighted_distances: np.ndarray) -> float:
    """
    Bertz's molecular complexity index.

    Adds up the information content of the bonding pattern and that of the
    distribution of atom types. See Bertz, J. Am. Chem. Soc. 103, 3599-3601 (1981).
    Computes the same value as RDKit's ``GraphDescriptors.BertzCT``, which counts
    the connections one atom pair at a time in Python.
    """
    if props.num_atoms < 2:
        return 0.0

    symmetry_classes = _symmetry_classes(weighted_distances)
    # RDKit reads the order of an aromatic bond as 1.5 and that of every other bond
    # as its bond type, which coincides with the bond order for the common types
    bond_orders = np.where(props.bond_is_aromatic, 1.5, props.bond_types)

    connection_counts = _bertz_connection_counts(props, symmetry_classes, bond_orders)
    if not len(connection_counts):
        connection_counts = np.ones(1)
    total_connections = connection_counts.sum()

    atom_type_counts = np.bincount(props.atomic_nums)
    atom_type_counts = atom_type_counts[atom_type_counts > 0]

    connection_entropy = total_connections * (
        _information_entropy(connection_counts) + np.log2(total_connections)
    )
    atom_type_entropy = props.num_atoms * _information_entropy(atom_type_counts)
    return float(atom_type_entropy + connection_entropy)


def _information_entropy(counts: np.ndarray) -> np.floating:
    """Shannon entropy in bits of the distribution the counts describe."""
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    return -(nonzero * np.log2(nonzero)).sum()


def _average_exact_mol_wt(mol: Mol, exact_mol_wt: float) -> float:
    """
    Compute average exact molecular weight.

    The AMW descriptor is exact molecular weight divided by total atom count,
    including implicit hydrogens in the atom denominator.
    """
    return exact_mol_wt / rdMolDescriptors.CalcNumAtoms(mol)


def calc_rdkit_2d(
    mol_regular: Mol,
    props_regular: AtomicProperties,
    distance_matrix_regular: DistanceMatrix,
    estate_indices: np.ndarray,
) -> np.ndarray:
    """
    Compute 2D descriptors that map directly to RDKit descriptor functions.
    """
    # the E-state VSA descriptors below need the E-state indices, which RDKit
    # computes in Python and memoizes on the molecule under this name; they are
    # already known here, so they are handed over through that memo
    mol_regular._eStateIndices = estate_indices

    # the complexity index groups atoms by distances that weigh every bond by the
    # inverse of its bond order
    weighted_distances = DistanceMatrix(mol_regular, use_bond_orders=True).matrix

    exact_mol_wt = Descriptors.ExactMolWt(mol_regular)
    values = [
        safe_value(_balaban_j, props_regular, distance_matrix_regular.matrix),
        safe_value(_bertz_complexity, props_regular, weighted_distances),
        rdMolDescriptors.CalcNumHBA(mol_regular),
        rdMolDescriptors.CalcNumHBD(mol_regular),
        rdMolDescriptors.CalcLabuteASA(mol_regular),
        *_calc_moe_type_descriptors(mol_regular),
        Crippen.MolLogP(mol_regular),
        Crippen.MolMR(mol_regular),
        rdMolDescriptors.CalcTPSA(mol_regular),
        rdMolDescriptors.CalcTPSA(mol_regular, includeSandP=True),
        exact_mol_wt,
        safe_value(_average_exact_mol_wt, mol_regular, exact_mol_wt),
    ]

    return np.asarray(values, dtype=np.float32)


def calc_rdkit_3d(mol_with_3d_conformer: Mol) -> np.ndarray:
    """
    Compute 3D descriptors that map directly to RDKit descriptor functions.
    """
    values = [
        safe_value(rdMolDescriptors.CalcPMI1, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPMI2, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPMI3, mol_with_3d_conformer),
        safe_value(rdMolDescriptors.CalcPBF, mol_with_3d_conformer),
    ]

    return np.asarray(values, dtype=np.float32)

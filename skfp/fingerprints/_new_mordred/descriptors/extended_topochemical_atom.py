import numpy as np
from rdkit import Chem
from rdkit.Chem import AddHs, Atom, Bond, BondType, Kekulize, Mol, RWMol, SanitizeMol

from skfp.fingerprints._new_mordred.utils.atomic_properties import _RDKIT_PERIODIC_TABLE
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.periodic_table import PERIOD

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

FEATURE_NAMES = ["WPath", "WPol"]


def calc(
    mol_kekulized: Mol,
    distance_matrix: DistanceMatrix,
    ring_count: int,
) -> tuple[np.ndarray, list[str]]:
    mol_alkane = build_alkane_mol(mol_kekulized)
    if not mol_alkane:
        pass  # TODO: handle errors

    distance_matrix_simplified = DistanceMatrix(mol_alkane)

    core_counts = np.array([get_core_count(atom) for atom in mol_kekulized.GetAtoms()])
    num_atoms = mol_kekulized.GetNumAtoms()

    eta_core_counts = get_eta_core_counts(core_counts, num_atoms)
    eta_shape_indices = get_eta_shape_indices(mol_kekulized, core_counts)
    eta_vem_counts = get_eta_vem_counts(mol_kekulized)
    eta_composite_indices = get_eta_composite_and_functionality_indices(
        mol_kekulized,
        distance_matrix.matrix,
        mol_alkane,
        distance_matrix_simplified.matrix,
    )
    eta_composite_index_non_local = eta_composite_indices[0]
    eta_composite_index_non_local_alkane = eta_composite_indices[4]

    values = np.concatenate([eta_core_counts], dtype=np.float32)
    return values, FEATURE_NAMES


def get_eta_core_counts(core_counts: np.ndarray, num_atoms: int) -> np.ndarray:
    """
    ETA core count descriptors, sum and average.
    """
    core_count_sum = core_counts.sum()
    return np.array([core_count_sum, core_count_sum / num_atoms])


def get_eta_shape_indices(mol: Mol, core_counts: np.ndarray) -> np.ndarray:
    """
    ETA shape indices, like core count sum but only for given degrees.
    """
    values = []
    for degree in [1, 3, 4]:
        value = np.mean(
            [
                core_count
                for atom, core_count in zip(mol.GetAtoms(), core_counts, strict=True)
                if atom.GetDegree() == degree
            ]
        )
        values.append(value)

    return np.array(values)


def get_eta_vem_counts(mol: Mol) -> np.ndarray:
    """
    ETA VEM(valence electron mobile) count descriptors.
    """
    beta_sum = 0
    beta_s_sum = 0
    beta_ns_sum = 0
    beta_ns_d_sum = 0

    for atom in mol.GetAtoms():
        beta_s = get_eta_beta_sigma(atom) / 2
        beta_ns_d = get_eta_beta_delta(atom)
        beta_ns = get_eta_beta_non_sigma(atom) / 2 + beta_ns_d
        beta = beta_s + beta_ns

        beta_sum += beta
        beta_s_sum += beta_ns_d
        beta_ns_sum += beta_ns
        beta_ns_d_sum += beta_ns_d

    num_atoms = mol.GetNumAtoms()
    beta_avg = beta_sum / num_atoms
    beta_s_avg = beta_s_sum / num_atoms
    beta_ns_avg = beta_ns_sum / num_atoms
    beta_ns_d_avg = beta_ns_d_sum / num_atoms

    return np.array(
        [
            beta_sum,
            beta_avg,
            beta_s_sum,
            beta_s_avg,
            beta_ns_sum,
            beta_ns_avg,
            beta_ns_d_sum,
            beta_ns_d_avg,
        ],
        dtype=np.float32,
    )


def get_eta_composite_and_functionality_indices(
    mol: Mol,
    distance_matrix: np.ndarray,
    mol_alkane: Mol,
    distance_matrix_alkane: np.ndarray,
) -> np.ndarray:
    """
    ETA composite index descriptor.
    """
    gamma_vals = [get_eta_gamma(atom) for atom in mol.GetAtoms()]
    alkane_gamma_vals = [get_eta_gamma(atom) for atom in mol_alkane.GetAtoms()]

    num_atoms = mol.GetNumAtoms()
    num_atoms_alkane = mol_alkane.GetNumAtoms()

    def eta_composite_index(
        gamma: list[float], dists: np.ndarray, local: bool
    ) -> float:
        if local:
            checker = lambda r: r == 1
        else:
            checker = lambda r: r != 0

        return sum(
            sum(
                np.sqrt(gamma[i] * gamma[j] / r**2)
                for j, r in enumerate(row)
                if i < j and checker(r)
            )
            for i, row in enumerate(dists)
        )

    # composite indices
    regular_non_local = eta_composite_index(gamma_vals, distance_matrix, local=False)
    regular_non_local_avg = regular_non_local / num_atoms

    regular_local = eta_composite_index(gamma_vals, distance_matrix, local=True)
    regular_local_avg = regular_local / num_atoms

    alkane_non_local = eta_composite_index(
        alkane_gamma_vals, distance_matrix_alkane, local=False
    )
    alkane_non_local_avg = alkane_non_local / num_atoms_alkane

    alkane_local = eta_composite_index(
        alkane_gamma_vals, distance_matrix_alkane, local=True
    )
    alkane_local_avg = alkane_local / num_atoms_alkane

    # functionality indices
    functionality_index_non_local = alkane_non_local - regular_non_local
    functionality_index_non_local_avg = functionality_index_non_local / num_atoms

    functionality_index_local = alkane_local - regular_local
    functionality_index_local_avg = functionality_index_local / num_atoms

    return np.array(
        [
            regular_non_local,
            regular_non_local_avg,
            regular_local,
            regular_local_avg,
            alkane_non_local,
            alkane_non_local_avg,
            alkane_local,
            alkane_local_avg,
            functionality_index_non_local,
            functionality_index_non_local_avg,
            functionality_index_local,
            functionality_index_local_avg,
        ],
        dtype=np.float32,
    )


def get_core_count(atom: Atom) -> float:
    """
    Atomic core-count term (alpha) used as a building block of ETA indices.
    Reflects the relative number of non-valence (core) electrons, scaled by period.
    """
    Z = atom.GetAtomicNum()
    if Z == 1:
        return 0.0
    Zv = _RDKIT_PERIODIC_TABLE.GetNOuterElecs(Z)
    PN = PERIOD[Z]
    return (Z - Zv) / (Zv * (PN - 1))


def get_eta_epsilon(atom: Atom) -> float:
    """
    ETA electronegativity-like measure (epsilon) for a single atom.
    Differences in epsilon between bonded atoms encode bond polarity.
    """
    Zv = _RDKIT_PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
    return 0.3 * Zv - get_core_count(atom)


def get_eta_beta_sigma(atom: Atom) -> float:
    """
    Sigma-bond contribution to an atom's ETA beta index, summed over
    non-hydrogen neighbors and weighted by similarity of their epsilon values.
    """
    e = get_eta_epsilon(atom)
    return sum(
        0.5 if abs(get_eta_epsilon(a) - e) <= 0.3 else 0.75
        for a in atom.GetNeighbors()
        if a.GetAtomicNum() != 1
    )


def get_other_bond_atom(bond: Bond, atom: Atom) -> Atom:
    begin = bond.GetBeginAtom()
    if atom.GetIdx() != begin.GetIdx():
        return begin
    return bond.GetEndAtom()


def get_eta_nonsigma_contribute(bond: Bond) -> float:
    """
    Non-sigma (pi, aromatic) contribution of a single bond to the ETA beta index.
    Weighted by bond order, aromaticity, and the epsilon difference of its endpoints.
    """
    if bond.GetBondType() is Chem.BondType.SINGLE:
        return 0.0

    f = 1.0
    if bond.GetBondTypeAsDouble() == Chem.BondType.TRIPLE:
        f = 2.0

    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    dEps = abs(get_eta_epsilon(a) - get_eta_epsilon(b))

    if bond.GetIsAromatic():
        y = 2.0
    elif dEps > 0.3:
        y = 1.5
    else:
        y = 1.0

    return y * f


def get_eta_beta_delta(atom: Atom) -> float:
    """
    Lone-pair (delta) contribution to an atom's ETA beta index.
    Nonzero only for acyclic atoms with lone pairs adjacent to an aromatic neighbor.
    """
    if (
        atom.GetIsAromatic()
        or atom.IsInRing()
        or _RDKIT_PERIODIC_TABLE.GetNOuterElecs(atom.GetAtomicNum())
        - atom.GetTotalValence()
        <= 0
    ):
        return 0.0

    for b in atom.GetNeighbors():
        if b.GetIsAromatic():
            return 0.5

    return 0.0


def get_eta_beta_non_sigma(atom: Atom) -> float:
    """
    Total non-sigma (pi, aromatic) bond contribution to an atom's ETA beta index,
    summed over all bonds to non-hydrogen neighbors.
    """
    return sum(
        get_eta_nonsigma_contribute(b)
        for b in atom.GetBonds()
        if get_other_bond_atom(b, atom).GetAtomicNum() != 1
    )


def get_eta_gamma(atom: Atom) -> float:
    """
    ETA gamma index for an atom: core count divided by total beta contribution.
    Represents an atom's topochemical "hardness" in the ETA framework.
    """
    beta = (
        get_eta_beta_sigma(atom)
        + get_eta_beta_non_sigma(atom)
        + get_eta_beta_delta(atom)
    )
    if beta == 0:
        return np.nan
    return get_core_count(atom) / beta


def build_alkane_mol(
    mol: Mol, explicit_hydrogens: bool = False, saturated: bool = False
) -> Mol | None:
    """
    Build a simplified copy of input molecule.

    saturated=False gives a carbon skeleton, replacing heavy atoms with carbons
    and all bonds with single ones.

    saturated=True keeps atom elements and formal charges, carbon-carbon bonds
    become single, while bonds touching a heteroatom keep their original order.

    Input hydrogens are dropped and only re-added at the end when
    `explicit_hydrogens` is set.
    """
    new_mol = RWMol()
    old_to_new = {}

    # copy heavy atoms
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        if saturated:
            new_atom = Atom(atom.GetAtomicNum())
            new_atom.SetFormalCharge(atom.GetFormalCharge())
        else:
            new_atom = Atom(6)  # carbon

        old_to_new[atom.GetIdx()] = new_mol.AddAtom(new_atom)

    # copy bonds between kept atoms
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        if not saturated and (begin.GetDegree() > 4 or end.GetDegree() > 4):
            return None

        i = old_to_new.get(begin.GetIdx())
        j = old_to_new.get(end.GetIdx())
        if i is None or j is None:
            continue  # one end was a hydrogen

        keep_order = saturated and (
            begin.GetAtomicNum() != 6 or end.GetAtomicNum() != 6
        )
        new_mol.AddBond(i, j, bond.GetBondType() if keep_order else BondType.SINGLE)

    new_mol = new_mol.GetMol()
    if SanitizeMol(new_mol, catchErrors=True) != 0:
        return None

    if explicit_hydrogens:
        new_mol = AddHs(new_mol)

    Kekulize(new_mol)
    return new_mol

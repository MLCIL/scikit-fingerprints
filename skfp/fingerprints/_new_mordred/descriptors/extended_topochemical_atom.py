import numpy as np
from rdkit.Chem import AddHs, Atom, BondType, Kekulize, Mol, RWMol, SanitizeMol

from skfp.fingerprints._new_mordred.utils.atomic_properties import _RDKIT_PERIODIC_TABLE
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.periodic_table import PERIOD

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_GET_N_OUTER_ELECS = _RDKIT_PERIODIC_TABLE.GetNOuterElecs

FEATURE_NAMES = [
    "ETA_alpha",
    "AETA_alpha",
    "ETA_shape_p",
    "ETA_shape_y",
    "ETA_shape_x",
    "ETA_beta",
    "AETA_beta",
    "ETA_beta_s",
    "AETA_beta_s",
    "ETA_beta_ns",
    "AETA_beta_ns",
    "ETA_beta_ns_d",
    "AETA_beta_ns_d",
    "ETA_eta",
    "AETA_eta",
    "ETA_eta_L",
    "AETA_eta_L",
    "ETA_eta_R",
    "AETA_eta_R",
    "ETA_eta_RL",
    "AETA_eta_RL",
    "ETA_eta_F",
    "AETA_eta_F",
    "ETA_eta_FL",
    "AETA_eta_FL",
    "ETA_eta_B",
    "AETA_eta_B",
    "ETA_eta_BR",
    "AETA_eta_BR",
    "ETA_dAlpha_A",
    "ETA_dAlpha_B",
    "ETA_epsilon_1",
    "ETA_epsilon_2",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
    "ETA_dBeta",
    "AETA_dBeta",
    "ETA_psi_1",
    "ETA_dPsi_A",
    "ETA_dPsi_B",
]


def calc(
    mol_kekulized: Mol,
    distance_matrix: DistanceMatrix,
    mol_kekulized_hydrogens: Mol,
    ring_count: int,
    n_frags: int,
) -> tuple[np.ndarray, list[str]]:
    # ETA descriptors require a connected molecule
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32), FEATURE_NAMES

    num_atoms = mol_kekulized.GetNumAtoms()

    # atomic properties
    atomic_nums, core_counts, epsilons = _atom_properties(mol_kekulized)
    degrees = np.fromiter(
        (atom.GetDegree() for atom in mol_kekulized.GetAtoms()),
        dtype=np.int64,
        count=num_atoms,
    )
    gamma, beta_sigma, beta_non_sigma, beta_delta = _beta_and_gamma(
        mol_kekulized, atomic_nums, core_counts, epsilons
    )

    # reference variants of the molecule; each may fail to build (e.g. heavy atom
    # with degree > 4), in which case the descriptors depending on it become NaN
    mol_alkane = build_reference_mol(mol_kekulized)
    mol_alkane_hydrogens = build_reference_mol(mol_kekulized, explicit_hydrogens=True)
    mol_saturated = build_reference_mol(
        mol_kekulized, explicit_hydrogens=True, saturated=True
    )

    core_count = core_counts.sum()
    core_count_mean = core_count / num_atoms

    # ETA_shape_{p,y,x}: fraction of total core count from atoms of degree 1/3/4
    shape = np.array([core_counts[degrees == d].sum() / core_count for d in (1, 3, 4)])

    # ETA VEM counts (sums and averages of beta / beta_s / beta_ns / beta_ns_d)
    vem_beta_s = beta_sigma.sum() / 2.0
    vem_beta_ns = beta_non_sigma.sum() / 2.0 + beta_delta.sum()
    vem_beta_ns_d = beta_delta.sum()
    vem_beta = vem_beta_s + vem_beta_ns
    eta_vem_counts = np.array(
        [
            vem_beta,
            vem_beta / num_atoms,
            vem_beta_s,
            vem_beta_s / num_atoms,
            vem_beta_ns,
            vem_beta_ns / num_atoms,
            vem_beta_ns_d,
            vem_beta_ns_d / num_atoms,
        ],
        dtype=np.float32,
    )

    # composite + functionality indices
    if mol_alkane is None:
        gamma_alkane = None
        distance_matrix_alkane = None
    else:
        z_a, core_a, eps_a = _atom_properties(mol_alkane)
        gamma_alkane, *_ = _beta_and_gamma(mol_alkane, z_a, core_a, eps_a)
        distance_matrix_alkane = DistanceMatrix(mol_alkane).matrix

    eta_composite = _composite_and_functionality(
        gamma,
        distance_matrix.matrix,
        gamma_alkane,
        distance_matrix_alkane,
        num_atoms,
    )

    eta_branching = _branching_indices(eta_composite[6], ring_count, num_atoms)

    # ETA delta alpha
    if mol_alkane is None:
        eta_delta_alpha = np.array([np.nan, np.nan], dtype=np.float32)
    else:
        core_count_alkane = core_a.sum()
        d_a = max((core_count - core_count_alkane) / num_atoms, 0.0)
        d_b = max((core_count_alkane - core_count) / num_atoms, 0.0)
        eta_delta_alpha = np.array([d_a, d_b], dtype=np.float32)

    eta_epsilon_values = _epsilon_values(
        epsilons,
        mol_kekulized_hydrogens,
        mol_alkane_hydrogens,
        mol_saturated,
    )

    # ETA delta beta
    delta_beta = vem_beta_ns - vem_beta_s
    eta_delta_beta = np.array([delta_beta, delta_beta / num_atoms], dtype=np.float32)

    # ETA psi
    psi = core_count / (num_atoms * eta_epsilon_values[1])
    eta_psi = np.array(
        [psi, max(0.714 - psi, 0.0), max(psi - 0.714, 0.0)], dtype=np.float32
    )

    values = np.concatenate(
        [
            [core_count, core_count_mean],
            shape,
            eta_vem_counts,
            eta_composite,
            eta_branching,
            eta_delta_alpha,
            eta_epsilon_values,
            eta_delta_beta,
            eta_psi,
        ],
        dtype=np.float32,
    )
    return values, FEATURE_NAMES


def _atom_properties(mol: Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-atom arrays of (atomic number, core count alpha, epsilon).
    """
    num_atoms = mol.GetNumAtoms()
    atomic_nums = np.empty(num_atoms, dtype=np.int64)
    core_counts = np.empty(num_atoms, dtype=np.float64)
    epsilons = np.empty(num_atoms, dtype=np.float64)

    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        zv = _GET_N_OUTER_ELECS(z)
        alpha = 0.0 if z == 1 else (z - zv) / (zv * (PERIOD[z] - 1))
        atomic_nums[i] = z
        core_counts[i] = alpha
        epsilons[i] = 0.3 * zv - alpha

    return atomic_nums, core_counts, epsilons


def _beta_and_gamma(
    mol: Mol,
    atomic_nums: np.ndarray,
    core_counts: np.ndarray,
    epsilons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-atom sigma, non-sigma, and delta beta contributions and gamma.
    """
    num_atoms = mol.GetNumAtoms()
    beta_sigma = np.zeros(num_atoms, dtype=np.float64)
    beta_non_sigma = np.zeros(num_atoms, dtype=np.float64)

    for bond in mol.GetBonds():
        a = bond.GetBeginAtomIdx()
        b = bond.GetEndAtomIdx()
        za = atomic_nums[a]
        zb = atomic_nums[b]

        # sigma contribution: only between heavy-atom neighbors
        if za != 1 and zb != 1:
            weight = 0.5 if abs(epsilons[a] - epsilons[b]) <= 0.3 else 0.75
            beta_sigma[a] += weight
            beta_sigma[b] += weight

        # non-sigma (pi / aromatic) bond contribution
        contribution = _nonsigma_contribute(bond, epsilons)
        if contribution:
            if zb != 1:
                beta_non_sigma[a] += contribution
            if za != 1:
                beta_non_sigma[b] += contribution

    beta_delta = np.fromiter(
        (_beta_delta(atom) for atom in mol.GetAtoms()),
        dtype=np.float64,
        count=num_atoms,
    )

    beta = beta_sigma + beta_non_sigma + beta_delta
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(beta == 0.0, np.nan, core_counts / beta)

    return gamma, beta_sigma, beta_non_sigma, beta_delta


def _nonsigma_contribute(bond, epsilons: np.ndarray) -> float:
    """
    Non-sigma (pi, aromatic) contribution of a single bond to the ETA beta index.
    """
    if bond.GetBondType() is BondType.SINGLE:
        return 0.0

    f = 2.0 if bond.GetBondTypeAsDouble() == BondType.TRIPLE else 1.0

    if bond.GetIsAromatic():
        y = 2.0
    else:
        d_eps = abs(epsilons[bond.GetBeginAtomIdx()] - epsilons[bond.GetEndAtomIdx()])
        y = 1.5 if d_eps > 0.3 else 1.0

    return y * f


def _beta_delta(atom) -> float:
    """
    Lone-pair (delta) contribution: 0.5 for an acyclic atom with lone pairs that
    is adjacent to an aromatic neighbor, otherwise 0.
    """
    if atom.GetIsAromatic() or atom.IsInRing():
        return 0.0
    if _GET_N_OUTER_ELECS(atom.GetAtomicNum()) - atom.GetTotalValence() <= 0:
        return 0.0
    for neighbor in atom.GetNeighbors():
        if neighbor.GetIsAromatic():
            return 0.5
    return 0.0


def _composite_index_pair(gamma: np.ndarray, dists: np.ndarray) -> tuple[float, float]:
    """
    ETA composite index, returning (non_local, local) together.

    ``non_local = sum_{i<j} sqrt(gamma_i gamma_j / r_ij^2)`` and ``local`` is the
    same sum restricted to bonded pairs (``r_ij == 1``).
    """
    upper_i, upper_j = np.triu_indices(gamma.shape[0], k=1)
    d = dists[upper_i, upper_j]
    terms = np.sqrt(gamma[upper_i] * gamma[upper_j] / d**2)
    non_local = terms.sum()
    local = terms[d == 1.0].sum()
    return float(non_local), float(local)


def _composite_and_functionality(
    gamma: np.ndarray,
    distance_matrix: np.ndarray,
    gamma_alkane: np.ndarray | None,
    distance_matrix_alkane: np.ndarray | None,
    num_atoms: int,
) -> np.ndarray:
    """
    ETA composite and functionality index descriptors.

    Reference-alkane values (and the functionality indices derived from them)
    are NaN when the reference alkane could not be built.
    """
    regular_non_local, regular_local = _composite_index_pair(gamma, distance_matrix)
    regular_non_local_avg = regular_non_local / num_atoms
    regular_local_avg = regular_local / num_atoms

    if gamma_alkane is None:
        alkane_non_local = alkane_non_local_avg = np.nan
        alkane_local = alkane_local_avg = np.nan
    else:
        num_atoms_alkane = gamma_alkane.shape[0]
        alkane_non_local, alkane_local = _composite_index_pair(
            gamma_alkane, distance_matrix_alkane
        )
        alkane_non_local_avg = alkane_non_local / num_atoms_alkane
        alkane_local_avg = alkane_local / num_atoms_alkane

    functionality_non_local = alkane_non_local - regular_non_local
    functionality_local = alkane_local - regular_local

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
            functionality_non_local,
            functionality_non_local / num_atoms,
            functionality_local,
            functionality_local / num_atoms,
        ],
        dtype=np.float32,
    )


def _branching_indices(
    composite_index_alkane_local: float, ring_count: int, num_atoms: int
) -> np.ndarray:
    """
    ETA branching index descriptors.
    """
    if num_atoms <= 1:
        return np.full(4, np.nan, dtype=np.float32)

    if num_atoms == 2:
        reference_alkane_branching = 1.0
    else:
        reference_alkane_branching = np.sqrt(2) + 0.5 * (num_atoms - 3)

    non_ring = reference_alkane_branching - composite_index_alkane_local
    ring = non_ring + 0.086 * ring_count

    return np.array(
        [non_ring, non_ring / num_atoms, ring, ring / num_atoms], dtype=np.float32
    )


def _epsilon_values(
    epsilons: np.ndarray,
    mol_hydrogens: Mol,
    mol_alkane_hydrogens: Mol | None,
    mol_saturated: Mol | None,
) -> np.ndarray:
    """
    ETA epsilon and epsilon delta descriptors.

    Types 3 and 4 (and the epsilon deltas derived from them) are NaN when the
    respective reference molecule could not be built.
    """
    _, _, eps_hydrogens = _atom_properties(mol_hydrogens)

    eps_1 = eps_hydrogens.mean()
    eps_2 = epsilons.mean()
    eps_3 = (
        _atom_properties(mol_alkane_hydrogens)[2].mean()
        if mol_alkane_hydrogens is not None
        else np.nan
    )
    eps_4 = (
        _atom_properties(mol_saturated)[2].mean()
        if mol_saturated is not None
        else np.nan
    )

    # heavy atoms and hydrogens bonded to heteroatoms, on the H-explicit molecule
    keep = [
        atom.GetIdx()
        for atom in mol_hydrogens.GetAtoms()
        if atom.GetAtomicNum() != 1 or atom.GetNeighbors()[0].GetAtomicNum() != 6
    ]
    eps_5 = eps_hydrogens[keep].mean()

    return np.array(
        [
            eps_1,
            eps_2,
            eps_3,
            eps_4,
            eps_5,
            eps_1 - eps_3,
            eps_1 - eps_4,
            eps_3 - eps_4,
            eps_2 - eps_5,
        ],
        dtype=np.float32,
    )


def build_reference_mol(
    mol: Mol, explicit_hydrogens: bool = False, saturated: bool = False
) -> Mol | None:
    """
    Build an simplified reference analog of the molecule.

    saturated=False gives a carbon alkane-like skeleton, replacing heavy atoms
    with carbons and all bonds with single ones.

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

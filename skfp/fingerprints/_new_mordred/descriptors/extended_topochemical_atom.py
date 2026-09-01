import numpy as np
from rdkit.Chem import AddHs, Atom, BondType, Kekulize, Mol, RWMol, SanitizeMol

from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import (
    _N_OUTER_ELECS,
    _RDKIT_PERIODIC_TABLE,
    AtomicProperties,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    atoms_apply_func,
)
from skfp.fingerprints._new_mordred.utils.periodic_table import ELEMENT_PERIOD

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
    kekulized_bond_types: np.ndarray,
    props: AtomicProperties,
    props_hydrogens: AtomicProperties,
    distance_matrix: DistanceMatrix,
    rings: RingSets,
    n_frags: int,
) -> np.ndarray:
    """
    Compute extended topochemical atom (ETA) descriptors.

    Kekulization changes neither the atoms nor the skeleton, so the properties and
    the distances of the hydrogen-suppressed molecule apply to the kekulized one as
    well; only the bond types, which the beta indices read, are different.
    """
    # ETA descriptors require a connected molecule
    if n_frags != 1:
        return np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)

    num_atoms = props.num_atoms
    ring_count = rings.num_rings

    # atomic properties
    atomic_nums = props.atomic_nums
    core_counts, epsilons = _core_counts_and_epsilons(atomic_nums)
    degrees = props.degrees

    ring_atoms = np.zeros(num_atoms, dtype=bool)
    ring_atoms[[atom for ring in rings.simple_ring_atom_sets for atom in ring]] = True

    gamma, beta_sigma, beta_non_sigma, beta_delta = _beta_and_gamma(
        props, kekulized_bond_types, ring_atoms, core_counts, epsilons
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

    # composite + functionality indices, which compare the molecule with its alkane
    # reference: the same skeleton with every atom a carbon and every bond single
    gamma_alkane = _alkane_gamma(degrees)
    eta_composite = _composite_and_functionality(
        gamma,
        distance_matrix.matrix,
        gamma_alkane,
        # the alkane has the same skeleton, and therefore the same distances
        None if gamma_alkane is None else distance_matrix.matrix,
        num_atoms,
    )

    eta_branching = _branching_indices(eta_composite[6], ring_count, num_atoms)

    # ETA delta alpha
    if gamma_alkane is None:
        eta_delta_alpha = np.array([np.nan, np.nan], dtype=np.float32)
    else:
        core_count_alkane = num_atoms * _CARBON_CORE_COUNT
        d_a = max((core_count - core_count_alkane) / num_atoms, 0.0)
        d_b = max((core_count_alkane - core_count) / num_atoms, 0.0)
        eta_delta_alpha = np.array([d_a, d_b], dtype=np.float32)

    eta_epsilon_values = _epsilon_values(
        epsilons,
        props_hydrogens,
        _alkane_hydrogens_mean_epsilon(degrees) if gamma_alkane is not None else np.nan,
        _saturated_mean_epsilon(props, _BOND_ORDER_OF_TYPE[kekulized_bond_types]),
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
    return values


def _atom_properties(mol: Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return per-atom arrays of (atomic number, core count alpha, epsilon).
    """
    atomic_nums = atoms_apply_func(Atom.GetAtomicNum, mol, np.int32)
    return atomic_nums, *_core_counts_and_epsilons(atomic_nums)


def _core_counts_and_epsilons(atomic_nums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Core count alpha and epsilon of every atom, both of which follow from the
    atomic number alone.
    """
    outer_elecs = _N_OUTER_ELECS[atomic_nums]

    # hydrogens have no core electrons, so their alpha is 0 by definition (and the
    # general formula would divide by zero, since their period is 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        alphas = (atomic_nums - outer_elecs) / (
            outer_elecs * (ELEMENT_PERIOD.lookup(atomic_nums) - 1)
        )
    core_counts = np.where(atomic_nums == 1, 0.0, alphas)
    epsilons = 0.3 * outer_elecs - core_counts

    return core_counts.astype(np.float32), epsilons.astype(np.float32)


def _beta_and_gamma(
    props: AtomicProperties,
    bond_types: np.ndarray,
    ring_atoms: np.ndarray,
    core_counts: np.ndarray,
    epsilons: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute per-atom sigma, non-sigma, and delta beta contributions and gamma.

    The bond types are those of the kekulized molecule, where the aromatic bonds
    have become single and double ones while keeping their aromatic flag.
    """
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    is_hydrogen = props.is_hydrogen
    epsilon_gaps = np.abs(epsilons[begins] - epsilons[ends])

    # sigma contribution: only between heavy-atom neighbors
    between_heavy = ~is_hydrogen[begins] & ~is_hydrogen[ends]
    sigma_weights = np.where(epsilon_gaps <= 0.3, 0.5, 0.75) * between_heavy
    beta_sigma = _sum_over_bonds(props, sigma_weights, sigma_weights)

    # non-sigma (pi / aromatic) bond contribution, which an atom only takes from a
    # bond leading to a heavy atom

    # a triple bond holds two pi bonds, any other multiple bond one
    pi_bonds = np.where(bond_types == int(BondType.TRIPLE), 2.0, 1.0)
    # an aromatic bond counts double, and one between unlike atoms counts one and a half
    weights = np.where(
        props.bond_is_aromatic, 2.0, np.where(epsilon_gaps > 0.3, 1.5, 1.0)
    )
    non_sigma = np.where(bond_types == int(BondType.SINGLE), 0.0, weights * pi_bonds)

    beta_non_sigma = _sum_over_bonds(
        props, non_sigma * ~is_hydrogen[ends], non_sigma * ~is_hydrogen[begins]
    )

    beta_delta = _beta_delta(props, bond_types, ring_atoms)

    beta = beta_sigma + beta_non_sigma + beta_delta
    with np.errstate(divide="ignore", invalid="ignore"):
        gamma = np.where(beta == 0.0, np.nan, core_counts / beta)

    return gamma, beta_sigma, beta_non_sigma, beta_delta


def _sum_over_bonds(
    props: AtomicProperties, at_begin: np.ndarray, at_end: np.ndarray
) -> np.ndarray:
    """
    Gather per-bond contributions onto the atoms at either end of the bonds.
    """
    totals = np.bincount(
        props.bond_begin_idxs, weights=at_begin, minlength=props.num_atoms
    )
    totals += np.bincount(
        props.bond_end_idxs, weights=at_end, minlength=props.num_atoms
    )
    return totals.astype(np.float32)


def _beta_delta(
    props: AtomicProperties, bond_types: np.ndarray, ring_atoms: np.ndarray
) -> np.ndarray:
    """
    Lone-pair (delta) contribution: 0.5 for an acyclic atom with lone pairs that is
    adjacent to an aromatic neighbor, otherwise 0.
    """
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    # RDKit's total valence: the bonds an atom has, plus its hydrogens
    bond_orders = _BOND_ORDER_OF_TYPE[bond_types]
    valences = props.total_num_hs + _sum_over_bonds(props, bond_orders, bond_orders)
    has_lone_pairs = props.outer_electrons - valences > 0

    aromatic_neighbors = _sum_over_bonds(
        props, props.is_aromatic[ends], props.is_aromatic[begins]
    )
    eligible = (
        ~props.is_aromatic & ~ring_atoms & has_lone_pairs & (aromatic_neighbors > 0)
    )
    return np.where(eligible, 0.5, 0.0).astype(np.float32)


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
    props_hydrogens: AtomicProperties,
    alkane_hydrogens_mean_epsilon: float,
    saturated_mean_epsilon: float,
) -> np.ndarray:
    """
    ETA epsilon and epsilon delta descriptors.

    Types 3 and 4 (and the epsilon deltas derived from them) are NaN when the
    respective reference variant does not exist.
    """
    atomic_nums = props_hydrogens.atomic_nums
    eps_hydrogens = _core_counts_and_epsilons(atomic_nums)[1]

    eps_1 = eps_hydrogens.mean()
    eps_2 = epsilons.mean()
    eps_3 = alkane_hydrogens_mean_epsilon
    eps_4 = saturated_mean_epsilon

    # heavy atoms and hydrogens bonded to heteroatoms, on the H-explicit molecule
    is_hydrogen = props_hydrogens.is_hydrogen
    bonded_to_carbon = np.zeros(props_hydrogens.num_atoms, dtype=bool)
    begins, ends = props_hydrogens.bond_begin_idxs, props_hydrogens.bond_end_idxs
    bonded_to_carbon[begins] = atomic_nums[ends] == 6
    bonded_to_carbon[ends] |= atomic_nums[begins] == 6
    eps_5 = eps_hydrogens[~is_hydrogen | ~bonded_to_carbon].mean()

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


# the alkane reference replaces every heavy atom with a carbon, which caps the
# degree it can have, and every bond with a single one
_MAX_CARBON_DEGREE = 4

# bond order per bond type, for the kekulized bonds read above
_BOND_ORDER_OF_TYPE = np.full(max(int(t) for t in BondType.values) + 1, np.nan)
_BOND_ORDER_OF_TYPE[[int(BondType.SINGLE), int(BondType.DOUBLE)]] = [1.0, 2.0]
_BOND_ORDER_OF_TYPE[[int(BondType.TRIPLE), int(BondType.AROMATIC)]] = [3.0, 1.5]

# the valences RDKit allows each element, padded to a rectangle; an element with no
# allowed valence, such as a transition metal, takes no implicit hydrogens at all
_MAX_ALLOWED_VALENCES = 4
_ALLOWED_VALENCES = np.full((119, _MAX_ALLOWED_VALENCES), -np.inf)
_TAKES_HYDROGENS = np.zeros(119, dtype=bool)
# a positive charge widens the valence budget of the elements holding few outer
# electrons and narrows it for the ones holding many
_CHARGE_DIRECTION = np.zeros(119)
for _atomic_num in range(1, 119):
    _allowed = [v for v in _RDKIT_PERIODIC_TABLE.GetValenceList(_atomic_num) if v >= 0]
    _TAKES_HYDROGENS[_atomic_num] = bool(_allowed)
    _ALLOWED_VALENCES[_atomic_num, : len(_allowed)] = _allowed
    if _allowed:
        _ALLOWED_VALENCES[_atomic_num, len(_allowed) :] = np.inf
    _CHARGE_DIRECTION[_atomic_num] = (
        1.0 if _RDKIT_PERIODIC_TABLE.GetNOuterElecs(_atomic_num) >= 4 else -1.0
    )
_CARBON_CORE_COUNT, _CARBON_EPSILON = (
    value.item() for value in _core_counts_and_epsilons(np.array([6]))
)
_HYDROGEN_EPSILON = _core_counts_and_epsilons(np.array([1]))[1].item()


def _alkane_gamma(degrees: np.ndarray) -> np.ndarray | None:
    """
    Gamma of every atom of the alkane reference: the same skeleton with every atom
    a carbon and every bond single.

    All of its atoms have the same core count and the same epsilon, so every bond
    contributes half a beta unit to both of its atoms and gamma comes out as the
    reciprocal of the degree. ``None`` when the reference does not exist, which is
    the case exactly when some atom has more bonds than a carbon can have.
    """
    if degrees.max(initial=0) > _MAX_CARBON_DEGREE:
        return None

    with np.errstate(divide="ignore"):
        # a lone atom has no bonds and therefore no beta to divide by
        return np.where(degrees == 0, np.nan, _CARBON_CORE_COUNT / (0.5 * degrees))


def _saturated_mean_epsilon(props: AtomicProperties, bond_orders: np.ndarray) -> float:
    """
    Mean epsilon over the hydrogen-explicit saturated reference variant.

    That variant keeps every element and formal charge, turns carbon-carbon bonds
    into single ones and leaves the rest as they are, then fills the free valences
    with hydrogens. Epsilon depends only on the element, so only how many hydrogens
    end up being added matters, not where they go.
    """
    is_carbon = props.atomic_nums == 6
    begins, ends = props.bond_begin_idxs, props.bond_end_idxs
    orders = np.where(is_carbon[begins] & is_carbon[ends], 1.0, bond_orders)
    valences = np.bincount(
        begins, weights=orders, minlength=props.num_atoms
    ) + np.bincount(ends, weights=orders, minlength=props.num_atoms)

    num_hydrogens = _implicit_hydrogen_count(
        props.atomic_nums, props.formal_charges, valences
    )
    if num_hydrogens is None:
        return np.nan

    epsilons = _core_counts_and_epsilons(props.atomic_nums)[1]
    total = epsilons.sum() + num_hydrogens * _HYDROGEN_EPSILON
    return float(total / (props.num_atoms + num_hydrogens))


def _implicit_hydrogen_count(
    atomic_nums: np.ndarray, formal_charges: np.ndarray, valences: np.ndarray
) -> int | None:
    """
    How many hydrogens RDKit would add to fill the free valences, or None if some
    atom has more bonds than its element allows, where sanitization would fail.

    An atom is given the smallest of the valences its element allows that covers the
    bonds it already has, and the rest is filled with hydrogens. A charge shifts that
    budget: it adds capacity to the elements holding few outer electrons and takes it
    away from the ones holding many.
    """
    needed = valences - _CHARGE_DIRECTION[atomic_nums] * formal_charges
    allowed = _ALLOWED_VALENCES[atomic_nums]

    fits = allowed >= needed[:, np.newaxis]
    takes_hydrogens = _TAKES_HYDROGENS[atomic_nums]
    if not fits.any(axis=1)[takes_hydrogens].all():
        return None

    smallest_fitting = allowed[np.arange(len(allowed)), fits.argmax(axis=1)]
    counts = np.where(takes_hydrogens, smallest_fitting - needed, 0.0)
    return int(counts.sum())


def _alkane_hydrogens_mean_epsilon(degrees: np.ndarray) -> float:
    """
    Mean epsilon over the hydrogen-explicit alkane reference.

    Its carbons fill their remaining bonds with hydrogens, so only how many atoms of
    either element there are matters, not how they are arranged.
    """
    num_carbons = len(degrees)
    num_hydrogens = num_carbons * _MAX_CARBON_DEGREE - degrees.sum()
    total = num_carbons * _CARBON_EPSILON + num_hydrogens * _HYDROGEN_EPSILON
    return float(total / (num_carbons + num_hydrogens))


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

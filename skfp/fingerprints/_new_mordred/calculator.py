import numpy as np
from rdkit.Chem import AddHs, GetMolFrags, Mol

from skfp.fingerprints._new_mordred.descriptors import (
    abc_index,
    acid_base,
    adjacency_matrix,
    aromatic,
    atom_count,
    autocorrelation,
    barysz_matrix,
    bcut,
    bond_count,
    carbon_types,
    chi,
    cpsa,
    detour_matrix,
    distance_matrix,
    eccentric_connectivity_index,
    estate,
    extended_topochemical_atom,
    fragment_complexity,
    geometric_index,
    gravitational_index,
    molecular_distance_edge,
    morse,
    path_count,
    polarizability,
    rdkit_descriptors,
    ring_count,
    rotatable_bond,
    topological_charge,
    topological_index,
    vdw_volume_abc,
    vertex_adjacency_info,
    walk_count,
    wiener_index,
    zagreb_index,
)
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.feature_names import (
    ALL_FEATURE_NAMES,
    FEATURE_NAMES_2D,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import (
    AdjacencyMatrix,
    DistanceMatrix,
    DistanceMatrix3D,
)
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_FEATURE_NAME_TO_IDX_2D = {name: i for i, name in enumerate(FEATURE_NAMES_2D)}
_FEATURE_NAME_TO_IDX_ALL = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}

# every descriptor returns its own module-level, constant list of feature names, so
# the positions they map to in the output can be resolved once and reused, instead
# of looking up ~1800 names in a dict for every molecule
# the cache holds the name list itself as well, both to key on identity and to keep
# the list alive so that its id() cannot be reused by another object
_OUTPUT_IDXS_CACHE: dict[tuple[int, bool], tuple[list[str], np.ndarray]] = {}


def _output_idxs(
    feature_names: list[str], idx_map: dict[str, int], use_3D: bool
) -> np.ndarray:
    key = (id(feature_names), use_3D)
    cached = _OUTPUT_IDXS_CACHE.get(key)
    if cached is not None and cached[0] is feature_names:
        return cached[1]

    idxs = np.fromiter(
        (idx_map[name] for name in feature_names),
        dtype=np.intp,
        count=len(feature_names),
    )
    _OUTPUT_IDXS_CACHE[key] = (feature_names, idxs)
    return idxs


def get_feature_names(use_3d: bool) -> np.ndarray:
    return (
        np.asarray(ALL_FEATURE_NAMES, dtype=object)
        if use_3d
        else np.asarray(FEATURE_NAMES_2D, dtype=object)
    )


def compute(mol: Mol, use_3D: bool) -> np.ndarray:
    n_features = len(ALL_FEATURE_NAMES) if use_3D else len(FEATURE_NAMES_2D)
    idx_map = _FEATURE_NAME_TO_IDX_ALL if use_3D else _FEATURE_NAME_TO_IDX_2D
    result = np.full(n_features, np.nan, dtype=np.float32)

    # dependencies
    n_frags = len(GetMolFrags(mol))

    # classic, RDKit-standardized molecule
    mol_regular = preprocess_mol(mol)
    distance_matrix_regular = DistanceMatrix(mol_regular)
    adjacency_matrix_regular = AdjacencyMatrix(mol_regular)

    # per-atom property arrays
    props_regular = AtomicProperties(mol_regular)
    rings_regular = ring_count.RingSets(mol_regular, props_regular)
    # connected subgraphs, shared by the chi and path count descriptors
    subgraphs_regular = Subgraphs(mol_regular, props_regular)

    # hydrogen-explicit molecule; AddHs on the already-standardized molecule gives
    # the same result as re-running the full preprocessing, for half the cost.
    # Note that its hydrogens are freshly added and therefore carry no coordinates
    # and sit after every heavy atom, so the 3D section below builds its own
    # hydrogen-explicit molecule instead, with its own (possibly different) atom
    # numbering; per-atom arrays of the two are not interchangeable.
    mol_hydrogens = AddHs(mol_regular)
    distance_matrix_hydrogens = DistanceMatrix(mol_hydrogens)
    props_hydrogens = AtomicProperties(mol_hydrogens)
    gasteiger_charges_hydrogens = props_hydrogens.gasteiger_charges

    # cpsa_3d reuses cpsa_2d values
    cpsa_2d = cpsa.calc_2d(gasteiger_charges_hydrogens)

    # kekulized molecule (aromatic -> single/double bonds); kekulizing a copy of
    # the standardized molecule avoids a second round of hydrogen removal, and the
    # copy is required because Kekulize modifies the molecule in place
    mol_kekulized = preprocess_mol(Mol(mol_regular), kekulize=True, sanitize=False)
    distance_matrix_kekulized = DistanceMatrix(mol_kekulized)
    props_kekulized = AtomicProperties(mol_kekulized)
    mol_kekulized_hydrogens = AddHs(mol_kekulized)
    # kekulization does not change the ring perception, so the ring count of the
    # hydrogen-suppressed molecule can be reused here
    num_rings = rings_regular.num_rings

    # graph radius and diameter from the hydrogen-suppressed distance matrix
    graph_radius = distance_matrix_regular.radius
    graph_diameter = distance_matrix_regular.diameter

    # 2D descriptors
    descriptors_2d = [
        abc_index.calc(props_regular, distance_matrix_regular),
        walk_count.calc(mol_regular, adjacency_matrix_regular),
        path_count.calc(mol_regular, props_regular, subgraphs_regular),
        adjacency_matrix.calc(props_regular, n_frags, adjacency_matrix_regular),
        wiener_index.calc(mol_regular, distance_matrix_regular),
        zagreb_index.calc(mol_regular, adjacency_matrix_regular),
        acid_base.calc(mol_regular),
        autocorrelation.calc(props_hydrogens, distance_matrix_hydrogens),
        estate.calc(mol_regular),
        rdkit_descriptors.calc_rdkit_2d(mol_regular, distance_matrix_regular),
        atom_count.calc(mol_regular, props_regular),
        bond_count.calc(props_hydrogens, mol_kekulized_hydrogens),
        carbon_types.calc(mol_kekulized),
        rotatable_bond.calc(mol_regular),
        vertex_adjacency_info.calc(props_regular),
        ring_count.calc(rings_regular),
        vdw_volume_abc.calc(rings_regular, mol_hydrogens),
        topological_index.calc(graph_radius, graph_diameter),
        extended_topochemical_atom.calc(
            props_kekulized,
            distance_matrix_kekulized,
            mol_kekulized_hydrogens,
            num_rings,
            n_frags,
        ),
        barysz_matrix.calc(props_regular, n_frags),
        bcut.calc(props_regular, n_frags),
        aromatic.calc(props_regular),
        topological_charge.calc(adjacency_matrix_regular, distance_matrix_regular),
        cpsa_2d,
        polarizability.calc(props_hydrogens),
        chi.calc(props_regular, subgraphs_regular),
        fragment_complexity.calc(props_regular),
        eccentric_connectivity_index.calc(
            adjacency_matrix_regular, distance_matrix_regular
        ),
        distance_matrix.calc(props_regular, n_frags, distance_matrix_regular),
        detour_matrix.calc(props_regular, n_frags),
        molecular_distance_edge.calc(
            props_regular, adjacency_matrix_regular, distance_matrix_regular
        ),
    ]

    for values, feature_names in descriptors_2d:
        result[_output_idxs(feature_names, idx_map, use_3D)] = values

    # 3D descriptors
    if use_3D:
        mol_hydrogens_conformer = preprocess_mol(
            mol, explicit_hydrogens=True, sanitize=False
        )
        conf_id = mol_hydrogens_conformer.GetIntProp("conf_id")
        distance_matrix_3d = DistanceMatrix3D(mol_hydrogens_conformer, conf_id)
        props_hydrogens_conformer = AtomicProperties(mol_hydrogens_conformer)

        # mol_regular keeps the 3D conformer (RemoveHs preserves heavy-atom
        # coordinates), so it is the heavy-atom 3D molecule
        distance_matrix_3d_regular = DistanceMatrix3D(mol_regular, conf_id)
        adjacency_matrix_hydrogens_conformer = AdjacencyMatrix(mol_hydrogens_conformer)

        descriptors_3d: list = [
            morse.calc(props_hydrogens_conformer, distance_matrix_3d),
            rdkit_descriptors.calc_rdkit_3d(mol_hydrogens_conformer),
            # the charges must come from the conformer molecule, since CPSA pairs
            # them atom by atom with surface areas computed from that same molecule
            cpsa.calc_3d(
                mol_hydrogens_conformer,
                cpsa_2d,
                props_hydrogens_conformer.gasteiger_charges,
            ),
            gravitational_index.calc(
                props_regular,
                props_hydrogens_conformer,
                distance_matrix_3d_regular,
                distance_matrix_3d,
                adjacency_matrix_regular,
                adjacency_matrix_hydrogens_conformer,
            ),
            geometric_index.calc(distance_matrix_3d),
        ]

        for values, feature_names in descriptors_3d:
            result[_output_idxs(feature_names, idx_map, use_3D)] = values

    return result

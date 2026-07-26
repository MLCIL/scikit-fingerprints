import numpy as np
from rdkit.Chem import GetMolFrags, GetSymmSSSR, Mol

from skfp.fingerprints._new_mordred.descriptors import (
    abc_index,
    acid_base,
    adjacency_matrix,
    aromatic,
    atom_count,
    autocorrelation,
    barysz_matrix,
    bond_count,
    carbon_types,
    cpsa,
    detour_matrix,
    distance_matrix,
    eccentric_connectivity_index,
    estate,
    extended_topochemical_atom,
    fragment_complexity,
    geometric_index,
    morse,
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
from skfp.fingerprints._new_mordred.utils.atomic_properties import gasteiger_charges
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

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

_FEATURE_NAME_TO_IDX_2D = {name: i for i, name in enumerate(FEATURE_NAMES_2D)}
_FEATURE_NAME_TO_IDX_ALL = {name: i for i, name in enumerate(ALL_FEATURE_NAMES)}


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

    # hydrogen-explicit molecule
    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    distance_matrix_hydrogens = DistanceMatrix(mol_hydrogens)
    gasteiger_charges_hydrogens = gasteiger_charges(mol_hydrogens)

    # cpsa_3d reuses cpsa_2d values
    cpsa_2d = cpsa.calc_2d(gasteiger_charges_hydrogens)

    # kekulized molecule (aromatic -> single/double bonds)
    mol_kekulized = preprocess_mol(mol, kekulize=True)
    distance_matrix_kekulized = DistanceMatrix(mol_kekulized)
    mol_kekulized_hydrogens = preprocess_mol(
        mol, kekulize=True, explicit_hydrogens=True
    )
    num_rings = len(GetSymmSSSR(mol_kekulized))

    # graph radius and diameter from the hydrogen-suppressed distance matrix
    graph_radius = distance_matrix_regular.radius
    graph_diameter = distance_matrix_regular.diameter

    # 2D descriptors
    descriptors_2d = [
        abc_index.calc(mol_regular, distance_matrix_regular),
        walk_count.calc(mol_regular, adjacency_matrix_regular),
        adjacency_matrix.calc(mol_regular, n_frags, adjacency_matrix_regular),
        wiener_index.calc(mol_regular, distance_matrix_regular),
        zagreb_index.calc(mol_regular, adjacency_matrix_regular),
        acid_base.calc(mol_regular),
        autocorrelation.calc(mol_hydrogens, distance_matrix_hydrogens),
        estate.calc(mol_regular),
        rdkit_descriptors.calc_rdkit_2d(mol_regular, distance_matrix_regular),
        atom_count.calc(mol_regular),
        bond_count.calc(mol_hydrogens, mol_kekulized_hydrogens),
        carbon_types.calc(mol_kekulized),
        rotatable_bond.calc(mol_regular),
        vertex_adjacency_info.calc(mol_regular),
        ring_count.calc(mol_regular),
        vdw_volume_abc.calc(mol_regular, mol_hydrogens),
        topological_index.calc(graph_radius, graph_diameter),
        extended_topochemical_atom.calc(
            mol_kekulized,
            distance_matrix_kekulized,
            mol_kekulized_hydrogens,
            num_rings,
            n_frags,
        ),
        barysz_matrix.calc(mol_regular, n_frags),
        aromatic.calc(mol_regular),
        topological_charge.calc(adjacency_matrix_regular, distance_matrix_regular),
        cpsa_2d,
        polarizability.calc(mol_hydrogens),
        fragment_complexity.calc(mol_regular),
        eccentric_connectivity_index.calc(
            adjacency_matrix_regular, distance_matrix_regular
        ),
        distance_matrix.calc(mol_regular, n_frags, distance_matrix_regular),
        detour_matrix.calc(mol_regular, n_frags),
    ]

    for values, feature_names in descriptors_2d:
        result[[idx_map[n] for n in feature_names]] = values

    # 3D descriptors
    if use_3D:
        mol_hydrogens_conformer = preprocess_mol(
            mol, explicit_hydrogens=True, sanitize=False
        )
        conf_id = mol_hydrogens_conformer.GetIntProp("conf_id")
        distance_matrix_3d = DistanceMatrix3D(mol_hydrogens_conformer, conf_id)

        descriptors_3d: list = [
            morse.calc(mol_hydrogens_conformer, distance_matrix_3d),
            rdkit_descriptors.calc_rdkit_3d(mol_hydrogens_conformer),
            cpsa.calc_3d(mol_hydrogens_conformer, cpsa_2d, gasteiger_charges_hydrogens),
            geometric_index.calc(distance_matrix_3d),
        ]

        for values, feature_names in descriptors_3d:
            result[[idx_map[n] for n in feature_names]] = values

    return result

from types import ModuleType

import numpy as np
from rdkit.Chem import AddHs, GetMolFrags, GetSymmSSSR, Mol

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
    chi,
    constitutional,
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

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

MODULES_2D: list[ModuleType] = [
    abc_index,
    acid_base,
    adjacency_matrix,
    aromatic,
    atom_count,
    autocorrelation,
    barysz_matrix,
    bond_count,
    carbon_types,
    chi,
    constitutional,
    cpsa,  # both 2D and 3D
    detour_matrix,
    distance_matrix,
    eccentric_connectivity_index,
    estate,
    extended_topochemical_atom,
    fragment_complexity,
    molecular_distance_edge,
    path_count,
    polarizability,
    rdkit_descriptors,  # both 2D and 3D
    ring_count,
    rotatable_bond,
    topological_charge,
    topological_index,
    vdw_volume_abc,
    vertex_adjacency_info,
    walk_count,
    wiener_index,
    zagreb_index,
]

MODULES_3D: list[ModuleType] = [
    cpsa,  # both 2D and 3D
    geometric_index,
    gravitational_index,
    morse,
    rdkit_descriptors,  # both 2D and 3D
]


def _output_idxs(
    modules: list[ModuleType], use_3D: bool
) -> dict[ModuleType, np.ndarray]:
    """
    Positions each module's features occupy in the output, resolved once per module.

    The 3D output appends the 3D features to the 2D ones, so a 2D feature has the
    same position in both outputs and needs only one mapping.
    """
    name_to_idx = {name: idx for idx, name in enumerate(ALL_FEATURE_NAMES)}
    return {
        module: np.fromiter(
            (name_to_idx[name] for name in _get_module_feature_names(module, use_3D)),
            dtype=np.intp,
        )
        for module in modules
    }


def _get_module_feature_names(module: ModuleType, use_3D: bool) -> list[str]:
    """
    Feature names a descriptor module contributes to the 2D or the 3D output.

    Modules contributing to both split their names into ``FEATURE_NAMES_2D`` and
    ``FEATURE_NAMES_3D``; the rest declare a single ``FEATURE_NAMES``.
    """
    split_names = getattr(
        module, "FEATURE_NAMES_3D" if use_3D else "FEATURE_NAMES_2D", None
    )
    return split_names if split_names is not None else module.FEATURE_NAMES


_OUTPUT_IDXS_2D = _output_idxs(MODULES_2D, use_3D=False)
_OUTPUT_IDXS_3D = _output_idxs(MODULES_3D, use_3D=True)


def get_feature_names(use_3d: bool) -> np.ndarray:
    return (
        np.asarray(ALL_FEATURE_NAMES, dtype=object)
        if use_3d
        else np.asarray(FEATURE_NAMES_2D, dtype=object)
    )


def compute(mol: Mol, use_3D: bool) -> np.ndarray:
    n_features = len(ALL_FEATURE_NAMES) if use_3D else len(FEATURE_NAMES_2D)
    result = np.full(n_features, np.nan, dtype=np.float32)

    # dependencies
    n_frags = len(GetMolFrags(mol))

    # classic, RDKit-standardized molecule
    mol_regular = preprocess_mol(mol)
    distance_matrix_regular = DistanceMatrix.from_mol(mol_regular)
    adjacency_matrix_regular = AdjacencyMatrix(mol_regular)

    # per-atom property arrays, read from the molecule once and shared
    props_regular = AtomicProperties.from_mol(mol_regular)

    # hydrogen-explicit molecule
    # added hydrogens have no coordinates, so for 3D we build this separately
    # note that atom numberings are different for those molecules
    mol_hydrogens = AddHs(mol_regular)
    props_hydrogens = AtomicProperties.with_hydrogens_added(
        mol_hydrogens, props_regular
    )
    distance_matrix_hydrogens = DistanceMatrix.with_hydrogens_added(
        distance_matrix_regular, props_hydrogens
    )
    gasteiger_charges_hydrogens = props_hydrogens.gasteiger_charges

    # cpsa_3d reuses cpsa_2d values
    cpsa_2d = cpsa.calc_2d(gasteiger_charges_hydrogens)

    # kekulized molecule (aromatic -> single/double bonds)
    mol_kekulized = preprocess_mol(mol, kekulize=True)
    distance_matrix_kekulized = DistanceMatrix.from_mol(mol_kekulized)
    mol_kekulized_hydrogens = preprocess_mol(
        mol, kekulize=True, explicit_hydrogens=True
    )
    num_rings = len(GetSymmSSSR(mol_kekulized))

    # graph radius and diameter from the hydrogen-suppressed distance matrix
    graph_radius = distance_matrix_regular.radius
    graph_diameter = distance_matrix_regular.diameter

    # 2D descriptors
    descriptors_2d: dict[ModuleType, np.ndarray] = {
        abc_index: abc_index.calc(mol_regular, distance_matrix_regular),
        walk_count: walk_count.calc(mol_regular, adjacency_matrix_regular),
        path_count: path_count.calc(mol_regular),
        adjacency_matrix: adjacency_matrix.calc(
            mol_regular, n_frags, adjacency_matrix_regular
        ),
        wiener_index: wiener_index.calc(mol_regular, distance_matrix_regular),
        zagreb_index: zagreb_index.calc(mol_regular, adjacency_matrix_regular),
        acid_base: acid_base.calc(mol_regular),
        autocorrelation: autocorrelation.calc(mol_hydrogens, distance_matrix_hydrogens),
        estate: estate.calc(mol_regular),
        rdkit_descriptors: rdkit_descriptors.calc_rdkit_2d(
            mol_regular, distance_matrix_regular
        ),
        atom_count: atom_count.calc(mol_regular),
        bond_count: bond_count.calc(mol_hydrogens, mol_kekulized_hydrogens),
        carbon_types: carbon_types.calc(mol_kekulized),
        constitutional: constitutional.calc(mol_hydrogens),
        rotatable_bond: rotatable_bond.calc(mol_regular),
        vertex_adjacency_info: vertex_adjacency_info.calc(props_regular),
        ring_count: ring_count.calc(mol_regular),
        vdw_volume_abc: vdw_volume_abc.calc(mol_regular, mol_hydrogens),
        topological_index: topological_index.calc(graph_radius, graph_diameter),
        extended_topochemical_atom: extended_topochemical_atom.calc(
            mol_kekulized,
            distance_matrix_kekulized,
            mol_kekulized_hydrogens,
            num_rings,
            n_frags,
        ),
        barysz_matrix: barysz_matrix.calc(mol_regular, n_frags),
        aromatic: aromatic.calc(props_regular),
        topological_charge: topological_charge.calc(
            adjacency_matrix_regular, distance_matrix_regular
        ),
        cpsa: cpsa_2d,
        polarizability: polarizability.calc(props_hydrogens),
        chi: chi.calc(mol_regular),
        fragment_complexity: fragment_complexity.calc(props_regular),
        eccentric_connectivity_index: eccentric_connectivity_index.calc(
            adjacency_matrix_regular, distance_matrix_regular
        ),
        distance_matrix: distance_matrix.calc(
            mol_regular, n_frags, distance_matrix_regular
        ),
        detour_matrix: detour_matrix.calc(mol_regular, n_frags),
        molecular_distance_edge: molecular_distance_edge.calc(
            mol_regular, adjacency_matrix_regular, distance_matrix_regular
        ),
    }

    for module, values in descriptors_2d.items():
        result[_OUTPUT_IDXS_2D[module]] = values

    # 3D descriptors
    if use_3D:
        mol_hydrogens_conformer = preprocess_mol(
            mol, explicit_hydrogens=True, sanitize=False
        )
        conf_id = mol_hydrogens_conformer.GetIntProp("conf_id")
        distance_matrix_3d = DistanceMatrix3D(mol_hydrogens_conformer, conf_id)

        # mol_regular keeps the 3D conformer (RemoveHs preserves heavy-atom
        # coordinates), so it is the heavy-atom 3D molecule
        distance_matrix_3d_regular = DistanceMatrix3D(mol_regular, conf_id)
        adjacency_matrix_hydrogens_conformer = AdjacencyMatrix(mol_hydrogens_conformer)

        descriptors_3d: dict[ModuleType, np.ndarray] = {
            morse: morse.calc(mol_hydrogens_conformer, distance_matrix_3d),
            rdkit_descriptors: rdkit_descriptors.calc_rdkit_3d(mol_hydrogens_conformer),
            cpsa: cpsa.calc_3d(
                mol_hydrogens_conformer, cpsa_2d, gasteiger_charges_hydrogens
            ),
            gravitational_index: gravitational_index.calc(
                mol_regular,
                mol_hydrogens_conformer,
                distance_matrix_3d_regular,
                distance_matrix_3d,
                adjacency_matrix_regular,
                adjacency_matrix_hydrogens_conformer,
            ),
            geometric_index: geometric_index.calc(distance_matrix_3d),
        }

        for module, values in descriptors_3d.items():
            result[_OUTPUT_IDXS_3D[module]] = values

    return result

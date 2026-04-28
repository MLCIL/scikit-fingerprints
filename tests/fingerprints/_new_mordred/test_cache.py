import numpy as np
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache


def test_cache_prepares_2d_dependencies_eagerly():
    mol = Chem.MolFromSmiles("CCO")

    cache = MordredMolCache.from_mol(mol, use_3D=False)

    assert cache.original_mol is mol
    assert cache.use_3d is False
    assert cache.mol_with_hydrogens is None
    assert cache.n_frags == 1
    assert cache.mol_regular.GetNumAtoms() == 3
    assert cache.mol_kekulized.GetNumAtoms() == 3
    assert cache.distance_matrix_regular.matrix.shape == (3, 3)
    assert cache.adjacency_matrix_regular.order(1).shape == (3, 3)
    assert np.allclose(np.diag(cache.distance_matrix_regular.matrix), 0)
    assert len(cache.autocorrelation_gmats) == 9
    assert len(cache.autocorrelation_gsums) == 9
    assert cache.autocorrelation_weights["c"].shape == (3,)
    assert cache.autocorrelation_centered_weights["c"].shape == (3,)

    value_shapes = {
        "adjacency_matrix_values": (12,),
        "aromatic_values": (2,),
        "barysz_values": (104,),
        "bcut_values": (24,),
        "bond_count_values": (9,),
        "chi_values": (56,),
        "constitutional_values": (16,),
        "cpsa_2d_values": (2,),
        "cpsa_3d_values": (41,),
        "detour_matrix_values": (14,),
        "distance_matrix_values": (12,),
        "eccentric_connectivity_index_values": (1,),
        "estate_values": (316,),
        "extended_topochemical_atom_values": (45,),
        "fragment_complexity_values": (1,),
        "framework_values": (1,),
        "information_content_values": (42,),
        "kappa_shape_index_values": (3,),
        "lipinski_values": (2,),
        "logs_values": (1,),
        "mcgowan_volume_values": (1,),
        "molecular_distance_edge_values": (19,),
        "molecular_id_values": (12,),
        "path_count_values": (21,),
        "polarizability_values": (2,),
        "topological_charge_values": (21,),
        "topological_index_values": (4,),
        "vdw_volume_abc_values": (1,),
        "vertex_adjacency_information_values": (1,),
    }
    for attr, shape in value_shapes.items():
        assert getattr(cache, attr).shape == shape


def test_cache_prepares_3d_hydrogen_variant_when_requested():
    mol = Chem.MolFromSmiles("O")

    cache = MordredMolCache.from_mol(mol, use_3D=True)

    assert cache.use_3d is True
    assert cache.mol_with_hydrogens is not None
    assert cache.mol_regular.GetNumAtoms() == 1
    assert cache.mol_with_hydrogens.GetNumAtoms() == 3
    assert cache.geometrical_index_values.shape == (4,)
    assert cache.gravitational_index_values.shape == (4,)
    assert cache.morse_values.shape == (160,)

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
    assert cache.adjacency_matrix_values.shape == (12,)
    assert cache.distance_matrix_values.shape == (12,)
    assert cache.eccentric_connectivity_index_values.shape == (1,)
    assert cache.estate_values.shape == (316,)
    assert cache.extended_topochemical_atom_values.shape == (45,)
    assert cache.fragment_complexity_values.shape == (1,)
    assert cache.framework_values.shape == (1,)
    assert cache.information_content_values.shape == (42,)
    assert cache.kappa_shape_index_values.shape == (3,)
    assert cache.lipinski_values.shape == (2,)
    assert cache.logs_values.shape == (1,)
    assert cache.mcgowan_volume_values.shape == (1,)
    assert cache.aromatic_values.shape == (2,)
    assert np.allclose(np.diag(cache.distance_matrix_regular.matrix), 0)
    assert len(cache.autocorrelation_gmats) == 9
    assert len(cache.autocorrelation_gsums) == 9
    assert cache.autocorrelation_weights["c"].shape == (3,)
    assert cache.autocorrelation_centered_weights["c"].shape == (3,)
    assert cache.barysz_values.shape == (104,)
    assert cache.bcut_values.shape == (24,)
    assert cache.bond_count_values.shape == (9,)
    assert cache.chi_values.shape == (56,)
    assert cache.constitutional_values.shape == (16,)
    assert cache.cpsa_2d_values.shape == (2,)
    assert cache.cpsa_3d_values.shape == (41,)
    assert cache.detour_matrix_values.shape == (14,)


def test_cache_prepares_3d_hydrogen_variant_when_requested():
    mol = Chem.MolFromSmiles("O")

    cache = MordredMolCache.from_mol(mol, use_3D=True)

    assert cache.use_3d is True
    assert cache.mol_with_hydrogens is not None
    assert cache.mol_regular.GetNumAtoms() == 1
    assert cache.mol_with_hydrogens.GetNumAtoms() == 3
    assert cache.geometrical_index_values.shape == (4,)
    assert cache.gravitational_index_values.shape == (4,)

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem.PropertyMol import PropertyMol
from rdkit.Chem.rdMolDescriptors import GetUSR

from skfp.fingerprints import USRFingerprint


@pytest.fixture
def mols_conformers_3_plus_atoms(mols_conformers_list):
    return [mol for mol in mols_conformers_list if mol.GetNumAtoms() >= 3]


def test_usr_bit_fingerprint(mols_conformers_3_plus_atoms):
    usr_fp = USRFingerprint()
    X_skfp = usr_fp.transform(mols_conformers_3_plus_atoms)

    X_rdkit = np.array(
        [
            GetUSR(mol, confId=mol.GetIntProp("conf_id"))
            for mol in mols_conformers_3_plus_atoms
        ]
    )

    assert_allclose(X_skfp, X_rdkit, atol=1e-3)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 12))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_usr_bit_fingerprint_default_conf_id(mols_conformers_3_plus_atoms):
    mols = [PropertyMol(mol) for mol in mols_conformers_3_plus_atoms]
    for mol in mols:
        mol.ClearProp("conf_id")

    usr_fp = USRFingerprint()
    X_skfp = usr_fp.transform(mols)

    X_rdkit = np.array([GetUSR(mol, confId=0) for mol in mols])

    assert_allclose(X_skfp, X_rdkit, atol=1e-3)
    assert all(mol.GetIntProp("conf_id") == 0 for mol in mols)


def test_usr_bit_fingerprint_transform_x_y(mols_conformers_3_plus_atoms):
    labels = np.arange(len(mols_conformers_3_plus_atoms))

    usr_fp = USRFingerprint()
    X_skfp, y_skfp = usr_fp.transform_x_y(mols_conformers_3_plus_atoms, labels)

    X_rdkit = [
        GetUSR(mol, confId=mol.GetIntProp("conf_id"))
        for mol in mols_conformers_3_plus_atoms
    ]
    y_rdkit = labels

    assert_allclose(X_skfp, X_rdkit, atol=1e-3)
    assert_equal(X_skfp.shape, (len(mols_conformers_3_plus_atoms), 12))
    assert np.issubdtype(X_skfp.dtype, np.floating)
    assert_equal(y_skfp, y_rdkit)


def test_usr_ignore_errors(mols_conformers_list, mols_conformers_3_plus_atoms):
    usr_fp = USRFingerprint(errors="ignore", n_jobs=-1)
    X_skfp = usr_fp.transform(mols_conformers_list)
    X_skfp_3_plus_atoms = usr_fp.transform(mols_conformers_3_plus_atoms)

    assert_allclose(X_skfp, X_skfp_3_plus_atoms)


def test_usr_copy(mols_conformers_3_plus_atoms):
    labels = np.ones(len(mols_conformers_3_plus_atoms))
    usr_fp = USRFingerprint(errors="ignore", n_jobs=-1)
    fps, labels_out = usr_fp.transform_x_y(
        mols_conformers_3_plus_atoms, labels, copy=True
    )

    assert_equal(labels, labels_out)
    assert labels is not labels_out


def test_usr_feature_names():
    usr_fp = USRFingerprint()
    feature_names = usr_fp.get_feature_names_out()

    assert_equal(len(feature_names), usr_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "centroid_dists_mean")
    assert_equal(
        feature_names[-1], "farthest_atom_from_farthest_to_centroid_cubic_root"
    )

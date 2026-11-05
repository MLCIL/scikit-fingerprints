import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import AddHs, Mol, MolFromSmiles
from rdkit.Chem.AllChem import Compute2DCoords, EmbedMultipleConfs
from scipy.sparse import csr_array

pytest.importorskip("mordred")

from mordred import Calculator, descriptors

from skfp.fingerprints import MordredFingerprint
from skfp.fingerprints._new_mordred import mordred_community


def test_mordred_community_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_fp = MordredFingerprint(implementation="mordredcommunity", n_jobs=-1)
    X_skfp = mordred_fp.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_mordred = [calc(mol) for mol in smallest_mols_list]
    X_mordred = np.array(X_mordred, dtype=np.float32)

    assert_allclose(X_skfp, X_mordred, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_community_sparse_fingerprint(smallest_smiles_list, smallest_mols_list):
    mordred_fp = MordredFingerprint(
        implementation="mordredcommunity", sparse=True, n_jobs=-1
    )
    X_skfp = mordred_fp.transform(smallest_smiles_list)

    calc = Calculator(descriptors, ignore_3D=True)
    X_mordred = [calc(mol) for mol in smallest_mols_list]
    X_mordred = csr_array(X_mordred, dtype=np.float32)

    assert_allclose(X_skfp.data, X_mordred.data, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_community_3D_fingerprint(mols_conformers_list):
    mordred_fp = MordredFingerprint(
        use_3D=True, implementation="mordredcommunity", n_jobs=-1
    )
    X_skfp = mordred_fp.transform(mols_conformers_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_mordred = [calc(mol) for mol in mols_conformers_list]
    X_mordred = np.array(X_mordred, dtype=np.float32)

    assert_allclose(X_skfp, X_mordred, equal_nan=True)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_community_3D_sparse_fingerprint(mols_conformers_list):
    mordred_fp = MordredFingerprint(
        use_3D=True, implementation="mordredcommunity", sparse=True, n_jobs=-1
    )
    X_skfp = mordred_fp.transform(mols_conformers_list)

    calc = Calculator(descriptors, ignore_3D=False)
    X_mordred = [calc(mol) for mol in mols_conformers_list]
    X_mordred = csr_array(X_mordred, dtype=np.float32)

    assert_allclose(X_skfp.data, X_mordred.data, equal_nan=True)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_community_3D_without_3D_conformers(
    smallest_mols_list, mols_conformers_list
):
    # molecules without conformers, copies with a flat one, as read from 2D files, and
    # with conf_id pointing to a conformer they do not have
    flat_mols = [Mol(mol) for mol in smallest_mols_list]
    for mol in flat_mols:
        Compute2DCoords(mol)
    missing_conformer_mols = [Mol(mol) for mol in mols_conformers_list]
    for mol in missing_conformer_mols:
        mol.SetIntProp("conf_id", max(c.GetId() for c in mol.GetConformers()) + 1)
    mols = smallest_mols_list + flat_mols + missing_conformer_mols

    mordred_fp = MordredFingerprint(use_3D=True, implementation="mordredcommunity")
    X = mordred_fp.transform(mols)

    mordred_fp_2D = MordredFingerprint(implementation="mordredcommunity")
    X_2D = mordred_fp_2D.transform(mols)

    # mordredcommunity interleaves the 3D descriptors with the 2D ones
    is_2D = np.isin(
        mordred_fp.get_feature_names_out(), mordred_fp_2D.get_feature_names_out()
    )
    assert_allclose(X[:, is_2D], X_2D, equal_nan=True)
    assert np.isnan(X[:, ~is_2D]).all()


# Mordred returns the principal moments of inertia of this molecule as complex
# numbers with no imaginary part
@pytest.mark.filterwarnings("ignore::numpy.exceptions.ComplexWarning")
def test_mordred_community_3D_uses_the_conf_id_conformer():
    mol = AddHs(MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
    EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
    mol.SetIntProp("conf_id", 1)
    mordred_fp = MordredFingerprint(use_3D=True, implementation="mordredcommunity")
    X = mordred_fp.transform([mol])

    calc = Calculator(descriptors, ignore_3D=False)
    X_second = np.array([calc(mol, id=1)], dtype=np.float32)
    X_first = np.array([calc(mol, id=0)], dtype=np.float32)

    assert_allclose(X, X_second, equal_nan=True)
    assert not np.allclose(X, X_first, equal_nan=True)


def test_mordred_community_feature_names():
    mordred_fp = MordredFingerprint(implementation="mordredcommunity")
    feature_names_skfp = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_skfp), mordred_fp.n_features_out)
    assert_equal(len(feature_names_skfp), len(set(feature_names_skfp)))

    calc = Calculator(descriptors)
    feature_names_mordred = [str(d) for d in calc.descriptors if d.require_3D is False]

    assert_equal(feature_names_skfp, feature_names_mordred)


def test_mordred_community_3D_feature_names():
    mordred_fp = MordredFingerprint(use_3D=True, implementation="mordredcommunity")
    feature_names_skfp = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_skfp), mordred_fp.n_features_out)
    assert_equal(len(feature_names_skfp), len(set(feature_names_skfp)))

    calc = Calculator(descriptors, ignore_3D=False)
    feature_names_mordred = [str(d) for d in calc.descriptors]

    assert_equal(feature_names_skfp, feature_names_mordred)


def test_mordred_community_passes_timeout(monkeypatch):
    calculator_kwargs = []

    def calculator_spy(*args, **kwargs):
        calculator_kwargs.append(kwargs)
        return Calculator(*args, **kwargs)

    monkeypatch.setattr(mordred_community, "Calculator", calculator_spy)
    mordred_fp = MordredFingerprint(implementation="mordredcommunity", timeout=5.0)
    mordred_fp.transform(["CCO"])

    assert calculator_kwargs
    assert all(kwargs["config"] == {"timeout": 5.0} for kwargs in calculator_kwargs)


def test_mordred_community_warns_about_max_ring_system_rank():
    mordred_fp = MordredFingerprint(
        implementation="mordredcommunity", max_ring_system_rank=30
    )
    with pytest.warns(UserWarning, match="max_ring_system_rank is only used"):
        mordred_fp.transform(["CCO"])

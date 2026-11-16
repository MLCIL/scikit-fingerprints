import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.AllChem import EmbedMultipleConfs
from scipy.sparse import csr_array

pytest.importorskip("mordred")

from mordred import Calculator, descriptors

from skfp.fingerprints import MordredFingerprint
from skfp.fingerprints._mordred_implementations import mordred_community


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


def test_mordred_community_3D_without_conformers(smallest_mols_list):
    mordred_fp = MordredFingerprint(use_3D=True, implementation="mordredcommunity")

    with pytest.raises(TypeError, match="must have at least one conformation"):
        mordred_fp.transform(smallest_mols_list)


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

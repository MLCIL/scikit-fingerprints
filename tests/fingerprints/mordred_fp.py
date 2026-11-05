import pickle
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import Conformer, Mol, MolFromSmiles, RenumberAtoms
from rdkit.Chem.AllChem import Compute2DCoords, EmbedMolecule, EmbedMultipleConfs
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import MordredFingerprint
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
Tests that do not need the mordredcommunity optional dependency. Its implementation
is tested in mordred_fp_mordredcommunity.py, and compared with the built-in one in
mordred_fp_parity.py.
"""


def test_mordred_fingerprint(smallest_smiles_list):
    X_seq = MordredFingerprint().transform(smallest_smiles_list)
    X_skfp = MordredFingerprint(n_jobs=-1).transform(smallest_smiles_list)

    assert_allclose(X_skfp, X_seq, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_sparse_fingerprint(smallest_smiles_list):
    X_dense = MordredFingerprint(n_jobs=-1).transform(smallest_smiles_list)
    X_skfp = MordredFingerprint(sparse=True, n_jobs=-1).transform(smallest_smiles_list)

    assert_allclose(X_skfp.toarray(), X_dense, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_smiles_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_3D_fingerprint(mols_conformers_list):
    X_seq = MordredFingerprint(use_3D=True).transform(mols_conformers_list)
    X_skfp = MordredFingerprint(use_3D=True, n_jobs=-1).transform(mols_conformers_list)

    assert_allclose(X_skfp, X_seq, equal_nan=True)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_3D_sparse_fingerprint(mols_conformers_list):
    X_dense = MordredFingerprint(use_3D=True, n_jobs=-1).transform(mols_conformers_list)
    X_skfp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1).transform(
        mols_conformers_list
    )

    assert_allclose(X_skfp.toarray(), X_dense, equal_nan=True)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_3D_without_3D_conformers(smallest_mols_list):
    # molecules without conformers, and copies with a flat one, as read from 2D files
    flat_mols = [Mol(mol) for mol in smallest_mols_list]
    for mol in flat_mols:
        Compute2DCoords(mol)
    mols = smallest_mols_list + flat_mols

    X = MordredFingerprint(use_3D=True).transform(mols)
    X_2D = MordredFingerprint().transform(mols)

    # 3D descriptors are appended after the 2D ones
    assert_allclose(X[:, : X_2D.shape[1]], X_2D, equal_nan=True)
    assert np.isnan(X[:, X_2D.shape[1] :]).all()


def test_mordred_feature_names():
    mordred_fp = MordredFingerprint()
    feature_names = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names), mordred_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))


def test_mordred_3D_feature_names():
    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names = mordred_fp.get_feature_names_out()
    feature_names_2D = MordredFingerprint().get_feature_names_out()

    assert_equal(len(feature_names), mordred_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    # 3D descriptors are appended after the 2D ones
    assert_equal(feature_names[: len(feature_names_2D)], feature_names_2D)


def test_mordred_wrong_implementation():
    mordred_fp = MordredFingerprint(implementation="mordred")
    with pytest.raises(InvalidParameterError):
        mordred_fp.transform(["CCO"])
    with pytest.raises(ValueError, match="implementation must be"):
        mordred_fp.get_feature_names_out()


def test_mordred_does_not_import_mordredcommunity():
    # mordredcommunity is optional, so it must be imported only when chosen;
    # checked in a fresh interpreter, since other tests import it in this one
    code = (
        "import pickle, pkgutil, sys\n"
        "from sklearn import clone\n"
        "import skfp\n"
        "for module in pkgutil.iter_modules(skfp.__path__, 'skfp.'):\n"
        "    __import__(module.name)\n"
        "from skfp.fingerprints import MordredFingerprint\n"
        "community_fp = MordredFingerprint(implementation='mordredcommunity')\n"
        "pickle.loads(pickle.dumps(clone(community_fp)))\n"
        "MordredFingerprint().get_feature_names_out()\n"
        "MordredFingerprint(n_jobs=2).transform(['CCO', 'CCN'])\n"
        "assert 'mordred' not in sys.modules, 'mordred imported'\n"
    )
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_mordred_community_missing_dependency(monkeypatch):
    # with None in sys.modules, importing mordred raises ModuleNotFoundError, as if
    # it was not installed
    monkeypatch.setitem(sys.modules, "mordred", None)
    monkeypatch.delitem(
        sys.modules, "skfp.fingerprints._new_mordred.mordred_community", raising=False
    )
    mordred_fp = MordredFingerprint(implementation="mordredcommunity")

    message = r'pip install "scikit-fingerprints\[mordredcommunity\]"'
    with pytest.raises(ImportError, match=message):
        mordred_fp.transform(["CCO"])
    with pytest.raises(ImportError, match=message):
        mordred_fp.get_feature_names_out()


def test_mordred_community_other_import_error(monkeypatch):
    # only a missing mordred is solved by installing the extra, so other import
    # errors are raised unchanged
    monkeypatch.setitem(
        sys.modules, "skfp.fingerprints._new_mordred.mordred_community", None
    )
    mordred_fp = MordredFingerprint(implementation="mordredcommunity")

    with pytest.raises(ModuleNotFoundError, match="mordred_community"):
        mordred_fp.transform(["CCO"])


def test_mordred_pickle_keeps_params():
    mordred_fp = MordredFingerprint(timeout=5.0, max_ring_system_rank=10)
    unpickled_fp = pickle.loads(pickle.dumps(mordred_fp))

    assert unpickled_fp.get_params() == mordred_fp.get_params()


def test_mordred_unpickles_old_objects_with_mordredcommunity():
    # objects pickled before the built-in implementation existed have none of its
    # parameters, and computed descriptors with mordredcommunity
    old_fp = MordredFingerprint()
    for name in ["implementation", "timeout", "max_ring_system_rank"]:
        delattr(old_fp, name)

    unpickled_fp = pickle.loads(pickle.dumps(old_fp))

    expected_params = MordredFingerprint(implementation="mordredcommunity").get_params()
    assert unpickled_fp.get_params() == expected_params


@pytest.mark.parametrize("use_3D", [False, True])
@pytest.mark.parametrize(
    "smiles", ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)c1ccncc1", "CN1CCC[C@H]1c1cccnc1"]
)
def test_mordred_does_not_depend_on_hydrogen_numbering(smiles, use_3D):
    mol = _embedded(smiles)
    mordred_fp = MordredFingerprint(use_3D=use_3D)

    X = mordred_fp.transform([mol])
    X_renumbered = mordred_fp.transform([_hydrogens_moved_to_front(mol)])

    assert_allclose(X_renumbered, X, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_mordred_conformer_without_conf_id_property():
    # like the other 3D fingerprints, use the first conformer when conf_id is unset
    mol = _embedded("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    mol_without_property = Mol(mol)
    mol_without_property.ClearProp("conf_id")
    mordred_fp = MordredFingerprint(use_3D=True)

    X = mordred_fp.transform([mol])
    X_without_property = mordred_fp.transform([mol_without_property])

    assert_allclose(X_without_property, X, equal_nan=True)


def test_mordred_3D_uses_the_conf_id_conformer():
    mol = preprocess_mol(
        MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"), explicit_hydrogens=True
    )
    EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
    mol.SetIntProp("conf_id", 1)
    mordred_fp = MordredFingerprint(use_3D=True)

    X = mordred_fp.transform([mol])
    X_second = mordred_fp.transform([_only_conformer(mol, 1)])
    X_first = mordred_fp.transform([_only_conformer(mol, 0)])

    # every 3D descriptor comes from the conformer conf_id points to
    assert_allclose(X, X_second, rtol=1e-5, equal_nan=True)
    assert not np.allclose(X, X_first, rtol=1e-5, equal_nan=True)


def _embedded(smiles: str) -> Mol:
    mol = preprocess_mol(MolFromSmiles(smiles), explicit_hydrogens=True)
    mol.SetIntProp("conf_id", EmbedMolecule(mol, randomSeed=0))
    return mol


def _hydrogens_moved_to_front(mol: Mol) -> Mol:
    # the same molecule and conformer, with every hydrogen numbered first
    hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() != 1]
    renumbered = RenumberAtoms(mol, hydrogens + heavy)
    renumbered.SetIntProp("conf_id", mol.GetIntProp("conf_id"))
    return renumbered


def _only_conformer(mol: Mol, conf_id: int) -> Mol:
    # a copy holding just the given conformer, as conformer 0
    single = Mol(mol)
    single.RemoveAllConformers()
    conformer = Conformer(mol.GetConformer(conf_id))
    conformer.SetId(0)
    single.AddConformer(conformer, assignId=False)
    single.SetIntProp("conf_id", 0)
    return single

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import Conformer, Mol, MolFromSmiles, RenumberAtoms
from rdkit.Chem.AllChem import EmbedMolecule, EmbedMultipleConfs

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint
from skfp.fingerprints._new_mordred.descriptors.cpsa import FEATURE_NAMES_3D
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from tests.fingerprints._new_mordred.utils import _MORDRED_NAMES

# surface areas come from FreeSASA's Lee-Richards, with Mordred's radii, instead of
# Mordred's own algorithm and differ by up to several percent, so the charged partial
# surface area descriptors built on them are compared in
# tests/fingerprints/_new_mordred/cpsa.py
_SURFACE_AREA_FEATURES = frozenset(FEATURE_NAMES_3D)


def test_new_mordred_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_sparse_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list).toarray()

    mordred_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list).toarray()

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(mols_conformers_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_old = mordred_fp.transform(mols_conformers_list)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(mols_conformers_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list).toarray()

    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(mols_conformers_list).toarray()

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_dative_bonds():
    # Mordred counts a dative bond as single wherever it needs bond orders, and
    # toward the valence of the atom it points to only
    mols = [
        MolFromSmiles(smiles)
        for smiles in [
            "C[NH2]->[Cu]c1ccccc1",
            "CC(=O)->[Fe]",
            "N->[Pt](<-N)(Cl)Cl",
            "O=C1O->[Zn]<-OC(=O)C1",
            "c1ccncc1->[Ru](Cl)(Cl)<-n1ccccc1",
            "C[B](c1ccccc1)<-N",
            # donors next to an aromatic ring, whose ETA lone pairs depend on the
            # dative bond not adding to their valence
            "c1ccccc1[CH2]->[Cu]",
            "c1ccccc1[C](=O)->[Fe]",
        ]
    ]
    new_mordred_fp = NewMordredFingerprint()
    X_new = new_mordred_fp.transform(mols)

    mordred_fp = MordredFingerprint()
    X_old = mordred_fp.transform(mols)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-3),
    )


def test_new_mordred_dummy_atoms():
    # Mordred weights a dummy atom by its atomic number, 0
    mols = [
        MolFromSmiles(smiles)
        for smiles in ["*c1ccccc1", "*CC(=O)O", "C[Si](*)(C)C", "[*:1]CC[*:2]"]
    ]
    new_mordred_fp = NewMordredFingerprint()
    X_new = new_mordred_fp.transform(mols)

    mordred_fp = MordredFingerprint()
    X_old = mordred_fp.transform(mols)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-3),
    )


@pytest.mark.parametrize("use_3D", [False, True])
@pytest.mark.parametrize(
    "smiles", ["CC(C)Cc1ccc(cc1)C(C)C(=O)O", "OC(=O)c1ccncc1", "CN1CCC[C@H]1c1cccnc1"]
)
def test_new_mordred_does_not_depend_on_hydrogen_numbering(smiles, use_3D):
    mol = _embedded(smiles)
    new_mordred_fp = NewMordredFingerprint(use_3D=use_3D)

    X = new_mordred_fp.transform([mol])
    X_renumbered = new_mordred_fp.transform([_hydrogens_moved_to_front(mol)])

    assert_allclose(X_renumbered, X, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_new_mordred_conformer_without_conf_id_property():
    # like the other 3D fingerprints, use the first conformer when conf_id is unset
    mol = _embedded("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
    mol_without_property = Mol(mol)
    mol_without_property.ClearProp("conf_id")
    new_mordred_fp = NewMordredFingerprint(use_3D=True)

    X = new_mordred_fp.transform([mol])
    X_without_property = new_mordred_fp.transform([mol_without_property])

    assert_allclose(X_without_property, X, equal_nan=True)


def test_new_mordred_3D_uses_the_conf_id_conformer():
    mol = preprocess_mol(
        MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"), explicit_hydrogens=True
    )
    EmbedMultipleConfs(mol, numConfs=2, randomSeed=42)
    mol.SetIntProp("conf_id", 1)
    new_mordred_fp = NewMordredFingerprint(use_3D=True)

    X = new_mordred_fp.transform([mol])
    X_second = new_mordred_fp.transform([_only_conformer(mol, 1)])
    X_first = new_mordred_fp.transform([_only_conformer(mol, 0)])

    # every 3D descriptor comes from the conformer conf_id points to
    assert_allclose(X, X_second, rtol=1e-5, equal_nan=True)
    assert not np.allclose(X, X_first, rtol=1e-5, equal_nan=True)


@pytest.mark.xfail(
    strict=False,
    reason="LogEE fixed in skfp, mordred-community reference is buggy "
    "(https://github.com/JacksonBurns/mordred-community/issues/24). The divergence "
    "is log(1 + 1/EE), so large-EE features still match within atol and xpass; "
    "not strict.",
)
def test_new_mordred_logee_diverges(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    logee = [name for name in df_new.columns if "LogEE" in name]
    assert logee

    assert_allclose(df_new[logee], df_old[logee], equal_nan=True, atol=1e-3)


def test_new_mordred_feature_names():
    new_mordred_fp = NewMordredFingerprint()
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint()
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # sets, since the order does not matter; translated, since skfp exposes most
    # features under readable names instead of mordred's codes
    feature_names_new = {_MORDRED_NAMES.get(name, name) for name in feature_names_new}
    assert feature_names_new == set(feature_names_old)


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # sets, since the order does not matter; translated, since skfp exposes most
    # features under readable names instead of mordred's codes
    feature_names_new = {_MORDRED_NAMES.get(name, name) for name in feature_names_new}
    assert feature_names_new == set(feature_names_old)


def test_mordred_names_mapping_is_one_to_one():
    # the value tests invert the mapping, so two skfp names sharing one mordred name
    # would silently drop a column from the comparison
    assert len(set(_MORDRED_NAMES.values())) == len(_MORDRED_NAMES)


def _arrays_to_dataframes(
    X_new: np.ndarray,
    feature_names_new: np.ndarray,
    X_old: np.ndarray,
    feature_names_old: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skfp_names = {mordred: skfp for skfp, mordred in _MORDRED_NAMES.items()}
    df_new = pd.DataFrame(X_new, columns=feature_names_new)
    df_old = pd.DataFrame(
        X_old, columns=[skfp_names.get(name, name) for name in feature_names_old]
    )
    return df_new, df_old[df_new.columns]


def _get_features_to_compare(
    df_new: pd.DataFrame, excluded: frozenset[str] = frozenset()
) -> list[str]:
    # LogEE is our deliberate fix of an upstream formula, so it never matches
    return [
        name for name in df_new.columns if "LogEE" not in name and name not in excluded
    ]


def _mismatched_features_msg(
    df_new: pd.DataFrame, df_old: pd.DataFrame, atol: float
) -> str:
    # build an error message listing the feature names whose values
    # differ beyond the tolerance
    close = np.isclose(df_new, df_old, equal_nan=True, atol=atol).all(axis=0)
    mismatched_names = df_new.columns[~close]
    if mismatched_names.empty:
        return ""
    return "Mismatched feature names:\n" + "\n".join(mismatched_names)


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

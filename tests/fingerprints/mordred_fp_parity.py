import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal
from rdkit.Chem import MolFromSmiles

pytest.importorskip("mordred")

from skfp.fingerprints import MordredFingerprint
from skfp.fingerprints._new_mordred.descriptors.cpsa import FEATURE_NAMES_3D
from tests.fingerprints._new_mordred.utils import _MORDRED_NAMES

"""
Comparisons of the built-in implementation with mordredcommunity, which need the
mordredcommunity optional dependency.
"""

# surface areas come from FreeSASA's Shrake-Rupley, with Mordred's radii but fewer
# test points per atom than Mordred's own, and differ by up to several percent, so
# the charged partial surface area descriptors built on them are compared in
# tests/fingerprints/_new_mordred/cpsa.py
_SURFACE_AREA_FEATURES = frozenset(FEATURE_NAMES_3D)


def test_mordred_implementations_match(smallest_mols_list):
    skfp_fp = MordredFingerprint(n_jobs=-1)
    X_skfp = skfp_fp.transform(smallest_mols_list)

    community_fp = MordredFingerprint(implementation="mordredcommunity", n_jobs=-1)
    X_community = community_fp.transform(smallest_mols_list)

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-3
        ),
    )
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_implementations_match_sparse(smallest_mols_list):
    skfp_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_skfp = skfp_fp.transform(smallest_mols_list).toarray()

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", sparse=True, n_jobs=-1
    )
    X_community = community_fp.transform(smallest_mols_list).toarray()

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-3
        ),
    )
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 1613))
    assert X_skfp.dtype == np.float32


def test_mordred_implementations_match_3D(mols_conformers_list):
    skfp_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_skfp = skfp_fp.transform(mols_conformers_list)

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", use_3D=True, n_jobs=-1
    )
    X_community = community_fp.transform(mols_conformers_list)

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-2
        ),
    )
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_implementations_match_3D_sparse(mols_conformers_list):
    skfp_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_skfp = skfp_fp.transform(mols_conformers_list).toarray()

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", use_3D=True, sparse=True, n_jobs=-1
    )
    X_community = community_fp.transform(mols_conformers_list).toarray()

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-2
        ),
    )
    assert_equal(X_skfp.shape, (len(mols_conformers_list), 1826))
    assert X_skfp.dtype == np.float32


def test_mordred_implementations_match_dative_bonds():
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
    skfp_fp = MordredFingerprint()
    X_skfp = skfp_fp.transform(mols)

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    X_community = community_fp.transform(mols)

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-3
        ),
    )


def test_mordred_implementations_match_dummy_atoms():
    # Mordred weights a dummy atom by its atomic number, 0
    mols = [
        MolFromSmiles(smiles)
        for smiles in ["*c1ccccc1", "*CC(=O)O", "C[Si](*)(C)C", "[*:1]CC[*:2]"]
    ]
    skfp_fp = MordredFingerprint()
    X_skfp = skfp_fp.transform(mols)

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    X_community = community_fp.transform(mols)

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_skfp)
    assert_allclose(
        df_skfp[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_skfp[features], df_community[features], atol=1e-3
        ),
    )


@pytest.mark.xfail(
    strict=False,
    reason="LogEE fixed in skfp, mordred-community reference is buggy "
    "(https://github.com/JacksonBurns/mordred-community/issues/24). The divergence "
    "is log(1 + 1/EE), so large-EE features still match within atol and xpass; "
    "not strict.",
)
def test_mordred_implementations_logee_diverges(smallest_mols_list):
    skfp_fp = MordredFingerprint(n_jobs=-1)
    X_skfp = skfp_fp.transform(smallest_mols_list)

    community_fp = MordredFingerprint(implementation="mordredcommunity", n_jobs=-1)
    X_community = community_fp.transform(smallest_mols_list)

    df_skfp, df_community = _arrays_to_dataframes(
        X_skfp,
        skfp_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    logee = [name for name in df_skfp.columns if "LogEE" in name]
    assert logee

    assert_allclose(df_skfp[logee], df_community[logee], equal_nan=True, atol=1e-3)


def test_mordred_implementations_feature_names_match():
    skfp_fp = MordredFingerprint()
    feature_names_skfp = skfp_fp.get_feature_names_out()

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    feature_names_community = community_fp.get_feature_names_out()

    assert_equal(len(feature_names_skfp), skfp_fp.n_features_out)
    assert_equal(len(feature_names_skfp), len(set(feature_names_skfp)))

    # translated, since skfp exposes most features under readable names instead of
    # mordred's codes; both list the 2D descriptors in the same order
    feature_names_skfp = [_MORDRED_NAMES.get(name, name) for name in feature_names_skfp]
    assert_equal(feature_names_skfp, feature_names_community)


def test_mordred_implementations_3D_feature_names_match():
    skfp_fp = MordredFingerprint(use_3D=True)
    feature_names_skfp = skfp_fp.get_feature_names_out()

    community_fp = MordredFingerprint(implementation="mordredcommunity", use_3D=True)
    feature_names_community = community_fp.get_feature_names_out()

    assert_equal(len(feature_names_skfp), skfp_fp.n_features_out)
    assert_equal(len(feature_names_skfp), len(set(feature_names_skfp)))

    # sets, since 3D descriptors are in a different order; translated, since skfp
    # exposes most features under readable names instead of mordred's codes
    feature_names_skfp = {_MORDRED_NAMES.get(name, name) for name in feature_names_skfp}
    assert feature_names_skfp == set(feature_names_community)


def test_mordred_names_mapping_is_one_to_one():
    # the value tests invert the mapping, so two skfp names sharing one mordred name
    # would silently drop a column from the comparison
    assert len(set(_MORDRED_NAMES.values())) == len(_MORDRED_NAMES)


def _arrays_to_dataframes(
    X_skfp: np.ndarray,
    feature_names_skfp: np.ndarray,
    X_community: np.ndarray,
    feature_names_community: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skfp_names = {mordred: skfp for skfp, mordred in _MORDRED_NAMES.items()}
    df_skfp = pd.DataFrame(X_skfp, columns=feature_names_skfp)
    df_community = pd.DataFrame(
        X_community,
        columns=[skfp_names.get(name, name) for name in feature_names_community],
    )
    return df_skfp, df_community[df_skfp.columns]


def _get_features_to_compare(
    df_skfp: pd.DataFrame, excluded: frozenset[str] = frozenset()
) -> list[str]:
    # LogEE is our deliberate fix of an upstream formula, so it never matches
    return [
        name for name in df_skfp.columns if "LogEE" not in name and name not in excluded
    ]


def _mismatched_features_msg(
    df_skfp: pd.DataFrame, df_community: pd.DataFrame, atol: float
) -> str:
    # build an error message listing the feature names whose values
    # differ beyond the tolerance
    close = np.isclose(df_skfp, df_community, equal_nan=True, atol=atol).all(axis=0)
    mismatched_names = df_skfp.columns[~close]
    if mismatched_names.empty:
        return ""
    return "Mismatched feature names:\n" + "\n".join(mismatched_names)

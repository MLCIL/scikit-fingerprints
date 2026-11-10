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
Comparisons of the optimized implementation with mordredcommunity, which need the
mordredcommunity optional dependency.
"""

# surface areas come from FreeSASA's Shrake-Rupley, with Mordred's radii but fewer
# test points per atom than Mordred's own, and differ by up to several percent, so
# the charged partial surface area descriptors built on them are compared in
# tests/fingerprints/_new_mordred/cpsa.py
_SURFACE_AREA_FEATURES = frozenset(FEATURE_NAMES_3D)


def test_mordred_implementations_match(smallest_mols_list):
    optimized_fp = MordredFingerprint(n_jobs=-1)
    X_optimized = optimized_fp.transform(smallest_mols_list)

    community_fp = MordredFingerprint(implementation="mordredcommunity", n_jobs=-1)
    X_community = community_fp.transform(smallest_mols_list)

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-3
        ),
    )
    assert_equal(X_optimized.shape, (len(smallest_mols_list), 1613))
    assert X_optimized.dtype == np.float32


def test_mordred_implementations_match_sparse(smallest_mols_list):
    optimized_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_optimized = optimized_fp.transform(smallest_mols_list).toarray()

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", sparse=True, n_jobs=-1
    )
    X_community = community_fp.transform(smallest_mols_list).toarray()

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-3
        ),
    )
    assert_equal(X_optimized.shape, (len(smallest_mols_list), 1613))
    assert X_optimized.dtype == np.float32


def test_mordred_implementations_match_3D(mols_conformers_list):
    optimized_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_optimized = optimized_fp.transform(mols_conformers_list)

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", use_3D=True, n_jobs=-1
    )
    X_community = community_fp.transform(mols_conformers_list)

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-2
        ),
    )
    assert_equal(X_optimized.shape, (len(mols_conformers_list), 1826))
    assert X_optimized.dtype == np.float32


def test_mordred_implementations_match_3D_sparse(mols_conformers_list):
    optimized_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_optimized = optimized_fp.transform(mols_conformers_list).toarray()

    community_fp = MordredFingerprint(
        implementation="mordredcommunity", use_3D=True, sparse=True, n_jobs=-1
    )
    X_community = community_fp.transform(mols_conformers_list).toarray()

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized, _SURFACE_AREA_FEATURES)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-2
        ),
    )
    assert_equal(X_optimized.shape, (len(mols_conformers_list), 1826))
    assert X_optimized.dtype == np.float32


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
    optimized_fp = MordredFingerprint()
    X_optimized = optimized_fp.transform(mols)

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    X_community = community_fp.transform(mols)

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-3
        ),
    )


def test_mordred_implementations_match_dummy_atoms():
    # Mordred weights a dummy atom by its atomic number, 0
    mols = [
        MolFromSmiles(smiles)
        for smiles in ["*c1ccccc1", "*CC(=O)O", "C[Si](*)(C)C", "[*:1]CC[*:2]"]
    ]
    optimized_fp = MordredFingerprint()
    X_optimized = optimized_fp.transform(mols)

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    X_community = community_fp.transform(mols)

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_optimized)
    assert_allclose(
        df_optimized[features],
        df_community[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(
            df_optimized[features], df_community[features], atol=1e-3
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
    optimized_fp = MordredFingerprint(n_jobs=-1)
    X_optimized = optimized_fp.transform(smallest_mols_list)

    community_fp = MordredFingerprint(implementation="mordredcommunity", n_jobs=-1)
    X_community = community_fp.transform(smallest_mols_list)

    df_optimized, df_community = _arrays_to_dataframes(
        X_optimized,
        optimized_fp.get_feature_names_out(),
        X_community,
        community_fp.get_feature_names_out(),
    )
    logee = [name for name in df_optimized.columns if "LogEE" in name]
    assert logee

    assert_allclose(df_optimized[logee], df_community[logee], equal_nan=True, atol=1e-3)


def test_mordred_implementations_feature_names_match():
    optimized_fp = MordredFingerprint()
    feature_names_optimized = optimized_fp.get_feature_names_out()

    community_fp = MordredFingerprint(implementation="mordredcommunity")
    feature_names_community = community_fp.get_feature_names_out()

    assert_equal(len(feature_names_optimized), optimized_fp.n_features_out)
    assert_equal(len(feature_names_optimized), len(set(feature_names_optimized)))

    # translated, since the optimized implementation exposes most features under
    # readable names instead of mordred's codes; both list the 2D descriptors in the
    # same order
    feature_names_optimized = [
        _MORDRED_NAMES.get(name, name) for name in feature_names_optimized
    ]
    assert_equal(feature_names_optimized, feature_names_community)


def test_mordred_implementations_3D_feature_names_match():
    optimized_fp = MordredFingerprint(use_3D=True)
    feature_names_optimized = optimized_fp.get_feature_names_out()

    community_fp = MordredFingerprint(implementation="mordredcommunity", use_3D=True)
    feature_names_community = community_fp.get_feature_names_out()

    assert_equal(len(feature_names_optimized), optimized_fp.n_features_out)
    assert_equal(len(feature_names_optimized), len(set(feature_names_optimized)))

    # sets, since 3D descriptors are in a different order; translated, since the
    # optimized implementation exposes most features under readable names instead of
    # mordred's codes
    feature_names_optimized = {
        _MORDRED_NAMES.get(name, name) for name in feature_names_optimized
    }
    assert feature_names_optimized == set(feature_names_community)


def test_mordred_names_mapping_is_one_to_one():
    # the value tests invert the mapping, so two optimized names sharing one mordred
    # name would silently drop a column from the comparison
    assert len(set(_MORDRED_NAMES.values())) == len(_MORDRED_NAMES)


def _arrays_to_dataframes(
    X_optimized: np.ndarray,
    feature_names_optimized: np.ndarray,
    X_community: np.ndarray,
    feature_names_community: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    optimized_names = {
        mordred: optimized for optimized, mordred in _MORDRED_NAMES.items()
    }
    df_optimized = pd.DataFrame(X_optimized, columns=feature_names_optimized)
    df_community = pd.DataFrame(
        X_community,
        columns=[optimized_names.get(name, name) for name in feature_names_community],
    )
    return df_optimized, df_community[df_optimized.columns]


def _get_features_to_compare(
    df_optimized: pd.DataFrame, excluded: frozenset[str] = frozenset()
) -> list[str]:
    # LogEE is our deliberate fix of an upstream formula, so it never matches
    return [
        name
        for name in df_optimized.columns
        if "LogEE" not in name and name not in excluded
    ]


def _mismatched_features_msg(
    df_optimized: pd.DataFrame, df_community: pd.DataFrame, atol: float
) -> str:
    # build an error message listing the feature names whose values
    # differ beyond the tolerance
    close = np.isclose(df_optimized, df_community, equal_nan=True, atol=atol).all(
        axis=0
    )
    mismatched_names = df_optimized.columns[~close]
    if mismatched_names.empty:
        return ""
    return "Mismatched feature names:\n" + "\n".join(mismatched_names)

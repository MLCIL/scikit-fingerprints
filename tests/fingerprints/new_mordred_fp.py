import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_equal

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint
from tests.fingerprints._new_mordred.utils import _MORDRED_NAMES


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
    features = _get_features_to_compare(df_new, df_old)
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
    features = _get_features_to_compare(df_new, df_old)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new, df_old)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list).toarray()

    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list).toarray()

    df_new, df_old = _arrays_to_dataframes(
        X_new,
        new_mordred_fp.get_feature_names_out(),
        X_old,
        mordred_fp.get_feature_names_out(),
    )
    features = _get_features_to_compare(df_new, df_old)
    assert_allclose(
        df_new[features],
        df_old[features],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(df_new[features], df_old[features], atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


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

    # sets, since the order does not matter; translated, since some families are
    # exposed under names spelling out what mordred abbreviates
    feature_names_new = {_MORDRED_NAMES.get(name, name) for name in feature_names_new}
    diff_names = feature_names_new - set(feature_names_old)
    assert not diff_names


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # sets, since the order does not matter; translated, since some families are
    # exposed under names spelling out what mordred abbreviates
    feature_names_new = {_MORDRED_NAMES.get(name, name) for name in feature_names_new}
    diff_names = feature_names_new - set(feature_names_old)
    assert not diff_names


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


def _get_features_to_compare(df_new: pd.DataFrame, df_old: pd.DataFrame) -> list[str]:
    not_computed = (df_new.isna() | df_old.isna()).all()
    return [
        name
        for name in df_new.columns
        if not not_computed[name] and "LogEE" not in name
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

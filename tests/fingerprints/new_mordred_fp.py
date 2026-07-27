import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint


def _logee_mask(feature_names: np.ndarray) -> np.ndarray:
    # LogEE intentionally diverges from mordred-community, whose implementation
    # adds a spurious exp(-a) term and computes log(1 + sum(exp(lambda_i)))
    # instead of the documented log(sum(exp(lambda_i))). The divergence is
    # log(1 + 1/EE), so only small-EE features exceed atol.
    # See https://github.com/JacksonBurns/mordred-community/issues/24.
    return np.array(["LogEE" in name for name in feature_names])


def test_new_mordred_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    feature_names = new_mordred_fp.get_feature_names_out()
    # temporary mask - will be eventually removed; LogEE is also excluded, as it
    # intentionally diverges from mordred-community (see _logee_mask); it is
    # compared separately in test_new_mordred_logee_diverges.
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0) & ~_logee_mask(
        feature_names
    )
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_sparse_fingerprint(smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_mols_list)

    mordred_fp = MordredFingerprint(sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    X_new = X_new.toarray()
    X_old = X_old.toarray()

    feature_names = new_mordred_fp.get_feature_names_out()
    # temporary mask - will be eventually removed; LogEE is also excluded, as it
    # intentionally diverges from mordred-community (see _logee_mask); it is
    # compared separately in test_new_mordred_logee_diverges.
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0) & ~_logee_mask(
        feature_names
    )
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-3,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-3),
    )
    assert_equal(X_new.shape, (len(smallest_mols_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    feature_names = new_mordred_fp.get_feature_names_out()
    # temporary mask - will be eventually removed; LogEE is also excluded, as it
    # intentionally diverges from mordred-community (see _logee_mask); it is
    # compared separately in test_new_mordred_logee_diverges.
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0) & ~_logee_mask(
        feature_names
    )
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-2),
    )
    assert_equal(X_new.shape, (len(mols_conformers_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(mols_conformers_list, smallest_mols_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(mols_conformers_list)

    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_old = mordred_fp.transform(smallest_mols_list)

    X_new = X_new.toarray()
    X_old = X_old.toarray()

    feature_names = new_mordred_fp.get_feature_names_out()
    # temporary mask - will be eventually removed; LogEE is also excluded, as it
    # intentionally diverges from mordred-community (see _logee_mask); it is
    # compared separately in test_new_mordred_logee_diverges.
    mask = ~(np.isnan(X_new) | np.isnan(X_old)).all(axis=0) & ~_logee_mask(
        feature_names
    )
    assert_allclose(
        X_new[:, mask],
        X_old[:, mask],
        equal_nan=True,
        atol=1e-2,
        err_msg=_mismatched_features_msg(X_new, X_old, mask, feature_names, atol=1e-2),
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

    feature_names = new_mordred_fp.get_feature_names_out()
    logee = _logee_mask(feature_names)
    assert logee.any()

    assert_allclose(X_new[:, logee], X_old[:, logee], equal_nan=True, atol=1e-3)


def test_new_mordred_feature_names():
    new_mordred_fp = NewMordredFingerprint()
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint()
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # we exclude changed feature names
    changed_name = np.array(
        [
            "autocorr" in feature
            or "MoRSE" in feature
            or "BCUT" in feature
            or "Dz" in feature
            for feature in feature_names_new
        ]
    )
    # sets, since the order does not matter
    feature_names_new = set(feature_names_new[~changed_name])
    feature_names_old = set(feature_names_old[~changed_name])

    diff_names = feature_names_new - feature_names_old
    assert not diff_names


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    # we exclude changed feature names
    changed_name = np.array(
        [
            "autocorr" in feature
            or "MoRSE" in feature
            or "BCUT" in feature
            or "Dz" in feature
            for feature in feature_names_new
        ]
    )
    # sets, since the order does not matter
    feature_names_new = set(feature_names_new[~changed_name])
    feature_names_old = set(feature_names_old[~changed_name])

    diff_names = feature_names_new - feature_names_old
    assert not diff_names


def _mismatched_features_msg(
    X_new: np.ndarray,
    X_old: np.ndarray,
    mask: np.ndarray,
    feature_names: np.ndarray,
    atol: float,
) -> str:
    # build an error message listing the feature names whose values
    # differ beyond the tolerance
    close = np.isclose(X_new[:, mask], X_old[:, mask], equal_nan=True, atol=atol)
    mismatched_cols = ~close.all(axis=0)
    mismatched_names = feature_names[mask][mismatched_cols]
    if len(mismatched_names) == 0:
        return ""
    return "Mismatched feature names:\n" + "\n".join(mismatched_names)

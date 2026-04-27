import numpy as np
import pytest
from mordred import Autocorrelation
from numpy.testing import assert_allclose, assert_equal

from skfp.fingerprints import MordredFingerprint, NewMordredFingerprint

NO_EXPLICIT_H_BOND_COUNT_FEATURES = ["nBonds", "nBondsS", "nBondsKS"]
NO_EXPLICIT_H_CONSTITUTIONAL_FEATURES = [
    "SZ",
    "Sm",
    "Sv",
    "Sse",
    "Spe",
    "Sare",
    "Sp",
    "Si",
    "MZ",
    "Mm",
    "Mv",
    "Mse",
    "Mpe",
    "Mare",
    "Mp",
    "Mi",
]
NO_EXPLICIT_H_CPSA_FEATURES = ["RNCG", "RPCG"]
NO_EXPLICIT_H_ETA_FEATURES = [
    "ETA_epsilon_1",
    "ETA_epsilon_3",
    "ETA_epsilon_4",
    "ETA_epsilon_5",
    "ETA_dEpsilon_A",
    "ETA_dEpsilon_B",
    "ETA_dEpsilon_C",
    "ETA_dEpsilon_D",
]
NO_EXPLICIT_H_FRAMEWORK_FEATURES = ["fMF"]
NO_EXPLICIT_H_INFORMATION_CONTENT_FEATURES = [
    f"{prefix}{order}"
    for prefix in ("IC", "TIC", "SIC", "BIC", "CIC", "MIC", "ZMIC")
    for order in range(6)
]
NO_EXPLICIT_H_LIPINSKI_FEATURES = ["GhoseFilter"]


@pytest.fixture(autouse=True)
def no_explicit_hydrogen_autocorrelation(monkeypatch):
    monkeypatch.setattr(
        Autocorrelation.AutocorrelationBase,
        "explicit_hydrogens",
        False,
    )


def test_new_mordred_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    # Keep the old Mordred reference in-process so the Autocorrelation
    # no-explicit-hydrogen monkeypatch is applied.
    mordred_fp = MordredFingerprint(n_jobs=1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporary mask - will be eventually removed
    mask = _parity_mask(
        X_new,
        X_old,
        new_mordred_fp.get_feature_names_out(),
    )

    assert_allclose(X_new[mask], X_old[mask], equal_nan=True)
    assert_equal(X_new.shape, (len(smallest_smiles_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_sparse_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    # Keep the old Mordred reference in-process so the Autocorrelation
    # no-explicit-hydrogen monkeypatch is applied.
    mordred_fp = MordredFingerprint(sparse=True, n_jobs=1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    X_new_dense = X_new.toarray()
    X_old_dense = X_old.toarray()

    # temporary mask - will be eventually removed
    mask = _parity_mask(
        X_new_dense,
        X_old_dense,
        new_mordred_fp.get_feature_names_out(),
    )

    assert_allclose(X_new_dense[mask], X_old_dense[mask], equal_nan=True)
    assert_equal(X_new.shape, (len(smallest_smiles_list), 1613))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    # Keep the old Mordred reference in-process so the Autocorrelation
    # no-explicit-hydrogen monkeypatch is applied.
    mordred_fp = MordredFingerprint(use_3D=True, n_jobs=1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    # temporary mask - will be eventually removed
    mask = _parity_mask(
        X_new,
        X_old,
        new_mordred_fp.get_feature_names_out(),
    )

    assert_allclose(X_new[mask], X_old[mask], equal_nan=True)
    assert_equal(X_new.shape, (len(smallest_smiles_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_3D_sparse_fingerprint(smallest_smiles_list):
    new_mordred_fp = NewMordredFingerprint(use_3D=True, sparse=True, n_jobs=-1)
    X_new = new_mordred_fp.transform(smallest_smiles_list)

    # Keep the old Mordred reference in-process so the Autocorrelation
    # no-explicit-hydrogen monkeypatch is applied.
    mordred_fp = MordredFingerprint(use_3D=True, sparse=True, n_jobs=1)
    X_old = mordred_fp.transform(smallest_smiles_list)

    X_new_dense = X_new.toarray()
    X_old_dense = X_old.toarray()

    # temporary mask - will be eventually removed
    mask = _parity_mask(
        X_new_dense,
        X_old_dense,
        new_mordred_fp.get_feature_names_out(),
    )

    assert_allclose(X_new_dense[mask], X_old_dense[mask], equal_nan=True)
    assert_equal(X_new.shape, (len(smallest_smiles_list), 1826))
    assert X_new.dtype == np.float32


def test_new_mordred_feature_names():
    new_mordred_fp = NewMordredFingerprint()
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint()
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    assert_equal(feature_names_new, feature_names_old)


def test_new_mordred_3D_feature_names():
    new_mordred_fp = NewMordredFingerprint(use_3D=True)
    feature_names_new = new_mordred_fp.get_feature_names_out()

    mordred_fp = MordredFingerprint(use_3D=True)
    feature_names_old = mordred_fp.get_feature_names_out()

    assert_equal(len(feature_names_new), new_mordred_fp.n_features_out)
    assert_equal(len(feature_names_new), len(set(feature_names_new)))

    assert_equal(feature_names_new, feature_names_old)


def _parity_mask(X_new, X_old, feature_names):
    """
    Create the old-vs-new Mordred parity mask.

    New Mordred BondCount, Constitutional, charge-only CPSA, ETA epsilon,
    Framework, InformationContent, and GhoseFilter descriptors intentionally keep 2D
    molecules hydrogen-suppressed, while default Mordred adds explicit
    hydrogens for nBonds/nBondsS/nBondsKS, the Constitutional descriptors
    SZ/Sm/Sv/Sse/Spe/Sare/Sp/Si/MZ/Mm/Mv/Mse/Mpe/Mare/Mp/Mi, CPSA RNCG/RPCG,
    ETA epsilon descriptors except epsilon_2, the fMF denominator, and all
    42 IC/TIC/SIC/BIC/CIC/MIC/ZMIC descriptors. GhoseFilter can differ because
    its atom-count criterion uses the hydrogen-suppressed atom count here and
    the explicit-H atom count in default Mordred.
    """
    mask = ~(np.isnan(X_new) | np.isnan(X_old))
    for name in [
        *NO_EXPLICIT_H_BOND_COUNT_FEATURES,
        *NO_EXPLICIT_H_CONSTITUTIONAL_FEATURES,
        *NO_EXPLICIT_H_CPSA_FEATURES,
        *NO_EXPLICIT_H_ETA_FEATURES,
        *NO_EXPLICIT_H_FRAMEWORK_FEATURES,
        *NO_EXPLICIT_H_INFORMATION_CONTENT_FEATURES,
        *NO_EXPLICIT_H_LIPINSKI_FEATURES,
    ]:
        mask[:, feature_names.tolist().index(name)] = False

    return mask

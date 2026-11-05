import warnings

import numpy as np
from descriptastorus.descriptors import RDKit2D, RDKit2DNormalized
from numpy.testing import assert_allclose, assert_equal, assert_raises
from scipy.sparse import csr_array

from skfp.fingerprints import RDKit2DDescriptorsFingerprint
from skfp.fingerprints.rdkit_2d_desc import (
    _get_rdkit_2d_distributions,
    _get_rdkit_2d_molpile_cdf,
    _normalize_rdkit_2d,
    _normalize_rdkit_2d_molpile,
)
from skfp.utils import no_rdkit_logs


def test_rdkit_2d_desc_fingerprint(smallest_mols_list):
    rdkit_2d_desc_fp = RDKit2DDescriptorsFingerprint(n_jobs=-1)
    X_skfp = rdkit_2d_desc_fp.transform(smallest_mols_list)

    gen = RDKit2D()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = np.array(X_descriptastorus, dtype=np.float32)

    assert_allclose(X_skfp, X_descriptastorus, atol=1e-3, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 200))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_sparse_fingerprint(smallest_mols_list):
    rdkit_2d_desc_fp = RDKit2DDescriptorsFingerprint(sparse=True, n_jobs=-1)
    X_skfp = rdkit_2d_desc_fp.transform(smallest_mols_list)

    gen = RDKit2D()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = csr_array(X_descriptastorus)

    assert_allclose(X_skfp.data, X_descriptastorus.data, atol=1e-3, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 200))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_normalized_fingerprint(smallest_mols_list):
    rdkit_2d_desc_fp = RDKit2DDescriptorsFingerprint(normalized=True, n_jobs=-1)
    X_skfp = rdkit_2d_desc_fp.transform(smallest_mols_list)

    gen = RDKit2DNormalized()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = np.vstack(X_descriptastorus)

    assert_allclose(X_skfp, X_descriptastorus, atol=1e-3, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 200))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_normalized_sparse_fingerprint(smallest_mols_list):
    rdkit_2d_desc_fp = RDKit2DDescriptorsFingerprint(
        normalized=True, sparse=True, n_jobs=-1
    )
    X_skfp = rdkit_2d_desc_fp.transform(smallest_mols_list)

    gen = RDKit2DNormalized()
    with no_rdkit_logs():
        X_descriptastorus = [
            np.array(gen.calculateMol(mol, None)) for mol in smallest_mols_list
        ]
    X_descriptastorus = [np.clip(x, -2147483647, 2147483647) for x in X_descriptastorus]
    X_descriptastorus = csr_array(X_descriptastorus)

    assert_allclose(X_skfp.data, X_descriptastorus.data, atol=1e-3, equal_nan=True)
    assert_equal(X_skfp.shape, (len(smallest_mols_list), 200))
    assert np.issubdtype(X_skfp.dtype, np.floating)


def test_rdkit_2d_desc_feature_names():
    rdkit_2d_desc_fp = RDKit2DDescriptorsFingerprint()
    feature_names_skfp = rdkit_2d_desc_fp.get_feature_names_out()

    assert_equal(len(feature_names_skfp), rdkit_2d_desc_fp.n_features_out)
    assert_equal(len(feature_names_skfp), len(set(feature_names_skfp)))

    gen = RDKit2DNormalized()
    feature_names_rdkit = np.asarray([name for name, obj in gen.columns])

    assert np.array_equal(feature_names_skfp, feature_names_rdkit)


def test_rdkit_2d_desc_unbranched_alkane():
    from rdkit.Chem import MolFromSmiles

    mol = MolFromSmiles("CCCCCC")
    fp = RDKit2DDescriptorsFingerprint()

    feature_names = fp.get_feature_names_out()
    idx = list(feature_names).index("fr_unbrch_alkane")

    result = fp.transform([mol])[0, idx]

    assert_equal(result, 1)


def test_rdkit_2d_desc_cdf_values():
    result = _normalize_rdkit_2d(
        [100.0, 300.0, 500.0],
        ["ExactMolWt", "ExactMolWt", "ExactMolWt"],
    )

    assert_equal(result.shape, (3,))
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))
    assert np.all(np.diff(result) >= 0)


def test_rdkit_2d_desc_cdf_clipping():
    distributions = _get_rdkit_2d_distributions()
    _, _, min_val, max_val = distributions["ExactMolWt"]

    result = _normalize_rdkit_2d(
        [min_val - 1000, max_val + 1000],
        ["ExactMolWt", "ExactMolWt"],
    )
    expected = _normalize_rdkit_2d(
        [min_val, max_val],
        ["ExactMolWt", "ExactMolWt"],
    )

    assert_allclose(result, expected)


def test_rdkit_2d_desc_cdf_missing_values():
    result = _normalize_rdkit_2d(
        [None, 5.0],
        ["ExactMolWt", "unknown_descriptor"],
    )

    assert_equal(result, [0.0, 0.0])


def test_rdkit_2d_desc_normalization_argument(smallest_mols_list):
    mols = smallest_mols_list[:2]

    raw_default = RDKit2DDescriptorsFingerprint().transform(mols)
    raw_none = RDKit2DDescriptorsFingerprint(normalization="none").transform(mols)

    assert_allclose(raw_default, raw_none, equal_nan=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        old_fp = RDKit2DDescriptorsFingerprint(normalized=True)

    assert_equal(len(caught), 1)
    assert_equal(caught[0].category, FutureWarning)

    new_fp = RDKit2DDescriptorsFingerprint(normalization="descriptastorus")

    assert_allclose(
        old_fp.transform(mols),
        new_fp.transform(mols),
        equal_nan=True,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", FutureWarning)
        old_raw_fp = RDKit2DDescriptorsFingerprint(normalized=False)

    assert_equal(len(caught), 1)
    assert_equal(caught[0].category, FutureWarning)

    assert_allclose(
        old_raw_fp.transform(mols),
        raw_default,
        equal_nan=True,
    )


def test_rdkit_2d_desc_normalization_invalid_arguments():
    with assert_raises(ValueError):
        RDKit2DDescriptorsFingerprint(
            normalized=True,
            normalization="descriptastorus",
        )

    with assert_raises(ValueError):
        RDKit2DDescriptorsFingerprint(normalization="invalid").transform(["CCO"])


def test_rdkit_2d_desc_molpile_fingerprint(smallest_mols_list):
    fp = RDKit2DDescriptorsFingerprint(normalization="molpile")
    result = fp.transform(smallest_mols_list)

    assert_equal(result.shape, (len(smallest_mols_list), 200))
    assert_equal(result.dtype, np.float32)
    assert np.all(np.isfinite(result))
    assert np.all((result >= 0) & (result <= 1))


def test_rdkit_2d_desc_molpile_cdf():
    distributions = _get_rdkit_2d_molpile_cdf()

    assert_equal(len(distributions), 200)

    thresholds, probabilities = distributions["ExactMolWt"]

    assert np.all(np.diff(thresholds) > 0)
    assert np.all(np.diff(probabilities) > 0)

    result = _normalize_rdkit_2d_molpile(
        [thresholds[0] - 1, thresholds[0], thresholds[-1] + 1, None],
        ["ExactMolWt"] * 4,
    )

    assert_allclose(
        result,
        [0.0, probabilities[0], 1.0, 0.0],
    )

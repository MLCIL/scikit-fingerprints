import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import MHFPFingerprint


def test_mhfp_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="bit", n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(encoder, mols_list)
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = np.stack([np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit])
    X_rdkit = (X_rdkit > 0).astype(int)

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_mhfp_count_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="count", n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(encoder, mols_list)
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = np.stack([np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit])

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_mhfp_raw_hashes_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="raw_hashes", n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(encoder, mols_list)
    X_rdkit = np.array(X_rdkit)

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert np.issubdtype(X_skfp.dtype, np.integer)
    assert np.all(X_skfp >= 0)


def test_mhfp_sparse_bit_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="bit", sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(encoder, mols_list)
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = csr_array(
        [(np.bincount(x, minlength=mhfp_fp.fp_size) > 0) for x in X_rdkit],
        dtype=int,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_mhfp_sparse_count_fingerprint(smiles_list, mols_list):
    mhfp_fp = MHFPFingerprint(variant="count", sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = MHFPEncoder.EncodeMolsBulk(encoder, mols_list)
    X_rdkit = np.array(X_rdkit)
    X_rdkit = np.mod(X_rdkit, mhfp_fp.fp_size)
    X_rdkit = csr_array(
        [np.bincount(x, minlength=mhfp_fp.fp_size) for x in X_rdkit],
        dtype=int,
    )

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_mhfp_sparse_raw_hashes_fingerprint(smiles_list):
    mhfp_fp = MHFPFingerprint(variant="raw_hashes", sparse=True, n_jobs=-1)
    X_skfp = mhfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = encoder.EncodeSmilesBulk(smiles_list)
    X_rdkit = csr_array(np.array(X_rdkit))

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), mhfp_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data >= 0)


def test_mhfp_wrong_radii(smiles_list):
    mhfp_fp = MHFPFingerprint(min_radius=3, radius=2)
    with pytest.raises(InvalidParameterError):
        mhfp_fp.transform(smiles_list)


def test_mhfp_random_state_parameter(smiles_list):
    """Regression test for GitHub issue #531: random_state must be an explicit
    constructor parameter (not just inherited silently) so that it appears in
    repr() and can be retrieved via get_params()."""
    fp_default = MHFPFingerprint()
    fp_seeded = MHFPFingerprint(random_state=42)
    fp_none = MHFPFingerprint(random_state=None)

    # random_state must appear in get_params()
    assert "random_state" in fp_default.get_params()
    assert fp_default.get_params()["random_state"] == 0
    assert fp_seeded.get_params()["random_state"] == 42
    assert fp_none.get_params()["random_state"] is None

    # Two runs with the same seed must produce identical results
    X1 = MHFPFingerprint(random_state=0).transform(smiles_list)
    X2 = MHFPFingerprint(random_state=0).transform(smiles_list)
    assert_equal(X1, X2)

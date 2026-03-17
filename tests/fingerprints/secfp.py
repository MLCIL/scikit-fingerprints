import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem.rdMHFPFingerprint import MHFPEncoder
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import SECFPFingerprint


def test_secfp_fingerprint(smiles_list, mols_list):
    secfp_fp = SECFPFingerprint(n_jobs=-1)
    X_skfp = secfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = np.array([encoder.EncodeSECFPMol(x) for x in mols_list])

    assert_equal(X_skfp, X_rdkit)
    assert_equal(X_skfp.shape, (len(smiles_list), secfp_fp.fp_size))
    assert_equal(X_skfp.dtype, np.uint8)
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_secfp_sparse_fingerprint(smiles_list, mols_list):
    secfp_fp = SECFPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = secfp_fp.transform(smiles_list)

    encoder = MHFPEncoder(2048, 0)
    X_rdkit = csr_array([encoder.EncodeSECFPMol(x) for x in mols_list])

    assert_equal(X_skfp.data, X_rdkit.data)
    assert_equal(X_skfp.shape, (len(smiles_list), secfp_fp.fp_size))
    assert_equal(X_skfp.dtype, np.uint8)
    assert np.all(X_skfp.data == 1)


def test_secfp_wrong_radii(smiles_list):
    secfp_fp = SECFPFingerprint(min_radius=3, radius=2)
    with pytest.raises(InvalidParameterError):
        secfp_fp.transform(smiles_list)


def test_secfp_random_state_parameter(smiles_list):
    """Regression test for GitHub issue #531: random_state must be an explicit
    constructor parameter (not just inherited silently) so that it appears in
    repr() and can be retrieved via get_params()."""
    fp_default = SECFPFingerprint()
    fp_seeded = SECFPFingerprint(random_state=42)
    fp_none = SECFPFingerprint(random_state=None)

    # random_state must appear in get_params()
    assert "random_state" in fp_default.get_params()
    assert fp_default.get_params()["random_state"] == 0
    assert fp_seeded.get_params()["random_state"] == 42
    assert fp_none.get_params()["random_state"] is None

    # Two runs with the same seed must produce identical results
    X1 = SECFPFingerprint(random_state=0).transform(smiles_list)
    X2 = SECFPFingerprint(random_state=0).transform(smiles_list)
    assert_equal(X1, X2)

import numpy as np
import pytest
from numpy.testing import assert_array_equal

torch = pytest.importorskip("torch")

from skfp.fingerprints.neural import CLAMPFingerprint  # noqa: E402


def test_clamp_output_shape(smiles_list):
    fp = CLAMPFingerprint()
    X = fp.transform(smiles_list)
    assert X.shape == (len(smiles_list), 768)


def test_clamp_output_dtype(smiles_list):
    fp = CLAMPFingerprint()
    X = fp.transform(smiles_list)
    assert X.dtype == np.float32


def test_clamp_output_finite(smiles_list):
    fp = CLAMPFingerprint()
    X = fp.transform(smiles_list)
    assert np.all(np.isfinite(X))


def test_clamp_deterministic(smiles_list):
    fp = CLAMPFingerprint()
    X1 = fp.transform(smiles_list)
    X2 = fp.transform(smiles_list)
    assert_array_equal(X1, X2)


def test_clamp_mols_vs_smiles(smiles_list, mols_list):
    fp = CLAMPFingerprint()
    X_smiles = fp.transform(smiles_list)
    X_mols = fp.transform(mols_list)
    assert_array_equal(X_smiles, X_mols)


def test_clamp_parallel(smiles_list):
    fp_single = CLAMPFingerprint(n_jobs=1)
    fp_parallel = CLAMPFingerprint(n_jobs=-1)
    X1 = fp_single.transform(smiles_list)
    X2 = fp_parallel.transform(smiles_list)
    np.testing.assert_allclose(X1, X2, atol=1e-5)

import numpy as np
import pytest
import scipy.sparse
from numpy.testing import assert_equal
from sklearn.utils._param_validation import InvalidParameterError

from skfp.fingerprints import E3FPFingerprint


def test_e3fp_bit_fingerprint(mols_conformers_list):
    e3fp_fp = E3FPFingerprint(n_jobs=-1)
    X_skfp = e3fp_fp.transform(mols_conformers_list)

    X_e3fp = np.stack(
        [e3fp_fp._calculate_single_mol_fingerprint(mol) for mol in mols_conformers_list]
    )

    assert_equal(X_skfp, X_e3fp)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), e3fp_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(np.isin(X_skfp, [0, 1]))


def test_e3fp_count_fingerprint(mols_conformers_list):
    e3fp_fp = E3FPFingerprint(count=True, n_jobs=-1)
    X_skfp = e3fp_fp.transform(mols_conformers_list)

    X_e3fp = np.stack(
        [e3fp_fp._calculate_single_mol_fingerprint(mol) for mol in mols_conformers_list]
    )

    assert_equal(X_skfp, X_e3fp)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), e3fp_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp >= 0)


def test_e3fp_sparse_bit_fingerprint(mols_conformers_list):
    e3fp_fp = E3FPFingerprint(sparse=True, n_jobs=-1)
    X_skfp = e3fp_fp.transform(mols_conformers_list)

    X_e3fp = scipy.sparse.vstack(
        [e3fp_fp._calculate_single_mol_fingerprint(mol) for mol in mols_conformers_list]
    )

    assert_equal(X_skfp.data, X_e3fp.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), e3fp_fp.fp_size))
    assert X_skfp.dtype == np.uint8
    assert np.all(X_skfp.data == 1)


def test_e3fp_sparse_count_fingerprint(mols_conformers_list):
    e3fp_fp = E3FPFingerprint(sparse=True, count=True, n_jobs=-1)
    X_skfp = e3fp_fp.transform(mols_conformers_list)

    X_e3fp = scipy.sparse.vstack(
        [e3fp_fp._calculate_single_mol_fingerprint(mol) for mol in mols_conformers_list]
    )

    assert_equal(X_skfp.data, X_e3fp.data)
    assert_equal(X_skfp.shape, (len(mols_conformers_list), e3fp_fp.fp_size))
    assert X_skfp.dtype == np.uint32
    assert np.all(X_skfp.data > 0)


def test_wrong_n_bits_before_folding(mols_conformers_list):
    e3fp_fp = E3FPFingerprint(fp_size=2048, n_bits_before_folding=1024)
    with pytest.raises(InvalidParameterError):
        e3fp_fp.transform(mols_conformers_list)

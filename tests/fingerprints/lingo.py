import numpy as np
import pytest
from numpy.ma.testutils import assert_not_equal
from numpy.testing import assert_equal

from skfp.fingerprints import LingoFingerprint


def test_lingo_bit_fingerprint(smiles_list, mols_list):
    lingo_fp_seq = LingoFingerprint(n_jobs=1)
    lingo_fp_parallel = LingoFingerprint(n_jobs=-1)

    X_seq = lingo_fp_seq.transform(smiles_list)
    X_parallel = lingo_fp_parallel.transform(smiles_list)

    assert_equal(X_seq, X_parallel)
    assert_equal(X_seq.shape, (len(smiles_list), lingo_fp_seq.fp_size))
    assert X_seq.dtype == np.uint8
    assert np.all(np.isin(X_seq, [0, 1]))


def test_lingo_count_fingerprint(smiles_list, mols_list):
    lingo_fp_seq = LingoFingerprint(count=True, n_jobs=1)
    lingo_fp_parallel = LingoFingerprint(count=True, n_jobs=-1)

    X_seq = lingo_fp_seq.transform(smiles_list)
    X_parallel = lingo_fp_parallel.transform(smiles_list)

    assert_equal(X_seq, X_parallel)
    assert_equal(X_seq.shape, (len(smiles_list), lingo_fp_seq.fp_size))
    assert X_seq.dtype == np.uint32
    assert np.all(X_seq >= 0)


def test_lingo_sparse_bit_fingerprint(smiles_list, mols_list):
    lingo_fp_dense = LingoFingerprint(n_jobs=-1)
    lingo_fp_sparse = LingoFingerprint(sparse=True, n_jobs=-1)

    X_dense = lingo_fp_dense.transform(smiles_list)
    X_sparse = lingo_fp_sparse.transform(smiles_list)

    assert_equal(X_dense, X_sparse.toarray())
    assert X_sparse.dtype == np.uint8
    assert np.all(X_sparse.data == 1)


def test_lingo_sparse_count_fingerprint(smiles_list, mols_list):
    lingo_fp_dense = LingoFingerprint(count=True, n_jobs=-1)
    lingo_fp_sparse = LingoFingerprint(count=True, sparse=True, n_jobs=-1)

    X_dense = lingo_fp_dense.transform(smiles_list)
    X_sparse = lingo_fp_sparse.transform(smiles_list)

    assert_equal(X_dense, X_sparse.toarray())
    assert X_sparse.dtype == np.uint32
    assert np.all(X_sparse.data > 0)


@pytest.mark.parametrize(
    "smiles_1,smiles_2,equal",
    [
        # ring numbers are normalized to zeros, should be identical
        ("c1ccccc1", "c0ccccc0", True),
        ("c1ccccc1", "c2ccccc2", True),
        ("c34cc2cc1ccccc1cc2cc3cccc4", "c00cc0cc0ccccc0cc0cc0cccc0", True),
        # 2-digit ring numbers with % should be handled as above, e.g. %10
        ("c10ccccc10O", "c%10ccccc%10O", True),
        (
            "c1cc2cc3c4c5c(cc6ccc7cc8cc9c%10c8c8c7c6c5c8c5c%10c(c1C9N1CCOCCOCCOCCOCCOCC1)c2c45)C3",
            "c0cc0cc0c0c0c(cc0ccc0cc0cc0c0c0c0c0c0c0c0c0c0c(c0C0N0CCOCCOCCOCCOCCOCC0)c0c0)C0",
            True,
        ),
        # common 2-letter elements simplification: Cl -> L, Br -> R
        ("[Na+].[Cl-]", "[Na+].[L-]", True),
        ("ClN(Cl)Cl", "LN(L)L", True),
        ("BrC#N", "RC#N", True),
        ("BrCl", "RL", True),
        # charges in brackets [] shouldn't be touched and give different fingerprints
        ("[Cl+3]", "[Cl+2]", False),
        ("[12CH0]OOOO[Al-2]", "[12CH2]OOOO[Al-0]", False),
    ],
)
def test_smiles_normalization(smiles_1: str, smiles_2: str, equal: bool):
    fp = LingoFingerprint()
    fp_smi_1 = fp.transform([smiles_1])
    fp_smi_2 = fp.transform([smiles_2])
    if equal:
        assert_equal(fp_smi_1, fp_smi_2)
    else:
        assert_not_equal(fp_smi_1, fp_smi_2)

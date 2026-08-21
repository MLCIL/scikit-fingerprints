import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("torch")
pytest.importorskip("chemprop")

from skfp.fingerprints.neural.chemeleon import ChemeleonFingerprint


def test_chemeleon_output_basic_properties(smiles_list):
    fp = ChemeleonFingerprint()
    X = fp.transform(smiles_list)

    assert X.shape == (len(smiles_list), 2048)
    assert X.dtype == np.float32
    assert np.all(np.isfinite(X))


def test_chemeleon_reference_values():
    # Reference values were computed with the upstream `CheMeleonFingerprint`
    # class from `chemeleon_fingerprint.py` in the JacksonBurns/chemeleon
    # repository, using the same pretrained `chemeleon_mp.pt` checkpoint
    # (Zenodo record 15460715). Both pipelines use the same Chemprop
    # featurization and message passing, so they produce bit-identical
    # 2048-dimensional embeddings; any regression in our implementation
    # (architecture, weight loading, featurization) will therefore fail this
    # test.
    smiles = [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CC(C)CC(=O)O",  # isovaleric acid
        "CC(=O)Nc1ccc(O)cc1",  # paracetamol
        "CN1CCCC1c1cccnc1",  # nicotine
    ]

    X_skfp = ChemeleonFingerprint().transform(smiles)
    expected = _load_chemeleon_data_file()

    assert_allclose(X_skfp, expected, atol=1e-5)
    assert X_skfp.shape == (len(smiles), 2048)
    assert X_skfp.dtype == np.float32


def test_chemeleon_mols_vs_smiles_input_parity(smiles_list, mols_list):
    fp = ChemeleonFingerprint()
    X_smiles = fp.transform(smiles_list)
    X_mols = fp.transform(mols_list)

    assert_array_equal(X_smiles, X_mols)


def test_chemeleon_parallel_consistency(smiles_list):
    X_serial = ChemeleonFingerprint(n_jobs=1).transform(smiles_list)
    X_parallel = ChemeleonFingerprint(n_jobs=-1).transform(smiles_list)

    assert_allclose(X_serial, X_parallel, atol=1e-5)


def _load_chemeleon_data_file() -> np.ndarray:
    filename = "chemeleon_fp.npy"

    if "tests" in os.listdir():
        return np.load(os.path.join("tests", "fingerprints", "data", filename))
    if "fingerprints" in os.listdir():
        return np.load(os.path.join("fingerprints", "data", filename))
    if "data" in os.listdir():
        return np.load(os.path.join("data", filename))

    raise FileNotFoundError(f"File {filename} not found")

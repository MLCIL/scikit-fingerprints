import os

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

pytest.importorskip("torch")

from skfp.fingerprints.neural import CLAMPFingerprint


def test_clamp_output_basic_properties(smiles_list):
    fp = CLAMPFingerprint()
    X = fp.transform(smiles_list)

    assert X.shape == (len(smiles_list), 768)
    assert X.dtype == np.float32
    assert np.all(np.isfinite(X))


def test_clamp_reference_values():
    # Reference values were computed with the upstream
    # `clamp.models.pretrained.PretrainedCLAMP.encode_smiles(...)` pipeline
    # from the ml-jku/clamp repository, using the same pretrained checkpoint.
    # Molecules were chosen so that their unfolded RDKit fingerprints have no
    # folding collisions at fp_size=8192, which is the only known source of
    # divergence between our implementation and the upstream one (see the
    # `mhnreact.molutils.getFingerprint` collision-handling bug). For these
    # collision-free molecules the two pipelines produce bit-identical
    # 768-dimensional embeddings; any regression in our implementation
    # (architecture, weight loading, featurization) will therefore fail this
    # test.
    smiles = [
        "CCO",                          # ethanol
        "c1ccccc1",                     # benzene
        "CC(=O)O",                      # acetic acid
        "CC(C)CC(=O)O",                 # isovaleric acid
        "CC(=O)Nc1ccc(O)cc1",           # paracetamol
        "CN1CCCC1c1cccnc1",             # nicotine
    ]

    X_skfp = CLAMPFingerprint().transform(smiles)
    expected = _load_clamp_data_file()

    assert_allclose(X_skfp, expected, atol=1e-5)
    assert X_skfp.shape == (len(smiles), 768)
    assert X_skfp.dtype == np.float32


def test_clamp_mols_vs_smiles_input_parity(smiles_list, mols_list):
    fp = CLAMPFingerprint()
    X_smiles = fp.transform(smiles_list)
    X_mols = fp.transform(mols_list)

    assert_array_equal(X_smiles, X_mols)


def test_clamp_parallel_consistency(smiles_list):
    X_serial = CLAMPFingerprint(n_jobs=1).transform(smiles_list)
    X_parallel = CLAMPFingerprint(n_jobs=-1).transform(smiles_list)

    assert_allclose(X_serial, X_parallel, atol=1e-5)


def _load_clamp_data_file() -> np.ndarray:
    filename = "clamp_fp.npy"

    if "tests" in os.listdir():
        return np.load(os.path.join("tests", "fingerprints", "data", filename))
    if "fingerprints" in os.listdir():
        return np.load(os.path.join("fingerprints", "data", filename))
    if "data" in os.listdir():
        return np.load(os.path.join("data", filename))

    raise FileNotFoundError(f"File {filename} not found")

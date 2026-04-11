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
    smiles = ["CCO", "c1ccccc1", "CC(=O)Nc1ccc(O)cc1"]
    expected_first_8 = np.array(
        [
            [
                3.818319,
                0.081569,
                0.166510,
                -0.048345,
                -0.207287,
                -0.002937,
                -0.048548,
                -0.057630,
            ],
            [
                3.977164,
                -0.037663,
                0.113107,
                -0.017698,
                -0.243209,
                0.029728,
                0.109190,
                -0.227499,
            ],
            [
                2.587135,
                0.015432,
                0.012606,
                -0.038937,
                -0.059818,
                0.035576,
                -0.049819,
                -0.073488,
            ],
        ],
        dtype=np.float32,
    )
    expected_norms = np.array([6.829771, 7.023746, 4.808838], dtype=np.float32)

    X = CLAMPFingerprint().transform(smiles)

    assert_allclose(X[:, :8], expected_first_8, atol=1e-5)
    assert_allclose(np.linalg.norm(X, axis=1), expected_norms, atol=1e-4)


def test_clamp_mols_vs_smiles_input_parity(smiles_list, mols_list):
    fp = CLAMPFingerprint()
    X_smiles = fp.transform(smiles_list)
    X_mols = fp.transform(mols_list)

    assert_array_equal(X_smiles, X_mols)


def test_clamp_parallel_consistency(smiles_list):
    X_serial = CLAMPFingerprint(n_jobs=1).transform(smiles_list)
    X_parallel = CLAMPFingerprint(n_jobs=-1).transform(smiles_list)

    assert_allclose(X_serial, X_parallel, atol=1e-5)

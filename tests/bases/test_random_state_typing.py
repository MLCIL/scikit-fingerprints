import numpy as np
import pytest

from skfp.fingerprints import MAPFingerprint


@pytest.mark.parametrize(
    "random_state",
    [
        42,
        np.random.RandomState(42),
        np.random.Generator(np.random.PCG64(42)),
        None,
    ],
    ids=["int", "RandomState", "Generator", "None"],
)
def test_map_fp_random_state_types(smallest_smiles_list, random_state):
    """MAPFingerprint should accept int, RandomState, Generator, or None."""
    fp = MAPFingerprint(random_state=random_state, n_jobs=-1)
    X = fp.transform(smallest_smiles_list)
    assert X.shape[0] == len(smallest_smiles_list)

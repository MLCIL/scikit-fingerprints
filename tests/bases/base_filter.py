import numpy as np
import pytest
from numpy.testing import assert_equal
from sklearn.utils._param_validation import InvalidParameterError

from skfp.filters import LipinskiFilter

"""
We cannot test most of BaseFilter directly, as it is an abstract base class (ABC),
but its methods are used extensively by inheriting classes. Therefore, we use
inheriting classes as proxies.
"""


def test_base_is_always_fitted(smiles_list):
    filt = LipinskiFilter()
    assert filt.__sklearn_is_fitted__()


def test_base_transform_copy(smiles_list):
    filt = LipinskiFilter()
    filtered_smiles = filt.transform(smiles_list, copy=False)
    filtered_smiles_2 = filt.transform(smiles_list, copy=True)
    assert_equal(filtered_smiles, filtered_smiles_2)


def test_base_invalid_params(smiles_list):
    filt = LipinskiFilter(allow_one_violation=2)  # type: ignore
    with pytest.raises(InvalidParameterError):
        filt.transform(smiles_list)


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_base_verbose(n_jobs, smiles_list, capsys):
    filt = LipinskiFilter(n_jobs=n_jobs, verbose=True)
    filt.transform(smiles_list)

    output = capsys.readouterr().err
    assert "100%" in output
    assert "it/s" in output


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_indicators_are_ndarray(n_jobs, smiles_list):
    """Filter indicators must be np.ndarray regardless of n_jobs (#557).

    Previously the multi-job path returned a Python list because
    run_in_parallel(flatten_results=True) yields a list and the
    per-condition vstack only ran for the condition_indicators branch.
    """
    filt = LipinskiFilter(n_jobs=n_jobs, return_type="indicators")
    out = filt.transform(smiles_list)
    assert isinstance(out, np.ndarray)
    assert out.dtype == bool
    assert out.shape == (len(smiles_list),)

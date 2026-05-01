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


def test_base_parallel_filter_indicators_return_array(smiles_list):
    filt = LipinskiFilter(return_type="indicators", n_jobs=2, batch_size=1)

    filter_indicators = filt.transform(smiles_list)

    assert isinstance(filter_indicators, np.ndarray)
    assert np.issubdtype(filter_indicators.dtype, bool)

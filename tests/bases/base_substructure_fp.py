import numpy as np
import pytest
from numpy.testing import assert_equal
from scipy.sparse import csr_array
from sklearn.utils._param_validation import InvalidParameterError

from skfp.bases.base_substructure_fp import BaseSubstructureFingerprint


@pytest.fixture
def substructure_smiles_list() -> list[str]:
    return [
        "CCOC",
        "CCOCCO",
        "CC(=O)O",
        "CC(=O)OC=O",
        "CCCCCC",
    ]


@pytest.fixture
def patterns_smarts_list() -> list[str]:
    return [
        "[#6]-[#6]-[#8]",
        "[#6]-[#6]-[#8]",
        "[#6]=[#8]",
        "[#6]=[#8]",
        "[#6]-[#6]-[#6]",
        "[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1",
    ]


def test_substructure_count_fingerprint(
    patterns_smarts_list: list[str], substructure_smiles_list: list[str]
):
    fp = BaseSubstructureFingerprint(patterns_smarts_list, count=True)
    X_count = fp.transform(substructure_smiles_list)

    assert isinstance(X_count, np.ndarray)
    assert X_count.dtype == np.uint32
    expected_count = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [0, 0, 0, 0, 4, 0],
        ]
    )
    assert_equal(X_count, expected_count)


def test_substructure_bit_fingerprint(
    patterns_smarts_list: list[str], substructure_smiles_list: list[str]
):
    fp = BaseSubstructureFingerprint(patterns_smarts_list, count=False)
    X_bit = fp.transform(substructure_smiles_list)

    assert isinstance(X_bit, np.ndarray)
    assert X_bit.dtype == np.uint8
    expected_bit = np.array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    assert_equal(X_bit, expected_bit)


def test_substructure_sparse_count_fingerprint(
    patterns_smarts_list: list[str], substructure_smiles_list: list[str]
):
    fp = BaseSubstructureFingerprint(patterns_smarts_list, count=True, sparse=True)
    X_count = fp.transform(substructure_smiles_list)

    assert isinstance(X_count, csr_array)
    assert X_count.dtype == np.uint32
    expected_count = csr_array(
        [
            [1, 1, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 2, 2, 0, 0],
            [0, 0, 0, 0, 4, 0],
        ]
    )
    assert_equal(X_count.data, expected_count.data)


def test_substructure_sparse_bit_fingerprint(
    patterns_smarts_list: list[str], substructure_smiles_list: list[str]
):
    fp = BaseSubstructureFingerprint(patterns_smarts_list, count=False, sparse=True)
    X_bit = fp.transform(substructure_smiles_list)
    assert isinstance(X_bit, csr_array)
    assert X_bit.dtype == np.uint8
    expected_bit = csr_array(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    assert_equal(X_bit.data, expected_bit.data)


def test_parameter_constraints_enabled(
    patterns_smarts_list: list[str], substructure_smiles_list: list[str]
):
    fp = BaseSubstructureFingerprint(patterns_smarts_list, count=42)  # type: ignore

    with pytest.raises(InvalidParameterError) as error:
        fp.transform(substructure_smiles_list)

    assert str(error.value).startswith(
        "The 'count' parameter of BaseSubstructureFingerprint must be an instance of 'bool'"
    )


def test_non_smarts_pattern(substructure_smiles_list: list[str]):
    invalid_patterns = [1, True, "abc"]
    with pytest.raises(InvalidParameterError) as error:
        fp = BaseSubstructureFingerprint(invalid_patterns)  # type: ignore
        fp.transform(substructure_smiles_list)

    assert str(error.value).startswith(
        "The 'patterns' parameter must be a sequence of SMARTS patterns."
    )


def test_invalid_mol_pattern(substructure_smiles_list: list[str]):
    invalid_patterns = ["X"]
    with pytest.raises(InvalidParameterError) as error:
        fp = BaseSubstructureFingerprint(invalid_patterns)  # type: ignore
        fp.transform(substructure_smiles_list)

    assert str(error.value).startswith("Got invalid SMARTS pattern")


def test_empty_patterns_list(substructure_smiles_list: list[str]):
    empty_patterns = []  # type: ignore
    with pytest.raises(InvalidParameterError) as error:
        fp = BaseSubstructureFingerprint(empty_patterns)  # type: ignore
        fp.transform(substructure_smiles_list)

    assert str(error.value).startswith(
        "The 'patterns' parameter must be a non-empty list of SMARTS patterns."
    )


def test_base_substructure_fp_no_random_state():
    """Regression test for GitHub issue #531: BaseSubstructureFingerprint
    does not use random_state internally, so it should not expose it as a
    constructor parameter (to avoid misleading the user)."""
    fp = BaseSubstructureFingerprint(["[OH]"])
    params = fp.get_params()
    assert "random_state" not in params, (
        "BaseSubstructureFingerprint should not expose random_state since it "
        "does not use any randomness. Got params: %s" % list(params.keys())
    )

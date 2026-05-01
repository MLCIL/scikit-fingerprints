import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Mol

from skfp.filters import (
    BMSFilter,
    BrenkFilter,
    GlaxoFilter,
    InpharmaticaFilter,
    LINTFilter,
    MLSMRFilter,
    NIBRFilter,
    NIHFilter,
    PAINSFilter,
    SureChEMBLFilter,
    ZINCBasicFilter,
)


def _get_substructural_filter_classes() -> list[type]:
    return [
        BMSFilter,
        BrenkFilter,
        GlaxoFilter,
        InpharmaticaFilter,
        LINTFilter,
        MLSMRFilter,
        NIBRFilter,
        NIHFilter,
        PAINSFilter,
        SureChEMBLFilter,
        ZINCBasicFilter,
    ]


def _get_substructural_filter_classes_and_num_conditions() -> list[tuple[type, int]]:
    return [
        (BMSFilter, 180),
        (BrenkFilter, 105),
        (GlaxoFilter, 55),
        (InpharmaticaFilter, 91),
        (LINTFilter, 57),
        (MLSMRFilter, 116),
        (NIBRFilter, 337),
        (NIHFilter, 180),
        (PAINSFilter, 480),  # all filters, A+B+C
        (SureChEMBLFilter, 166),
        (ZINCBasicFilter, 50),
    ]


@pytest.mark.parametrize("substruct_filt_cls", _get_substructural_filter_classes())
def test_substructural_filter(substruct_filt_cls, mols_list):
    filt = substruct_filt_cls()
    mols_filtered = filt.transform(mols_list)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert len(mols_filtered) <= len(mols_list)


@pytest.mark.parametrize("substruct_filt_cls", _get_substructural_filter_classes())
def test_substructural_filter_parallel(substruct_filt_cls, smiles_list):
    filt = substruct_filt_cls()
    smiles_filtered_sequential = filt.transform(smiles_list)

    filt = substruct_filt_cls(n_jobs=-1)
    smiles_filtered_parallel = filt.transform(smiles_list)

    assert_equal(smiles_filtered_sequential, smiles_filtered_parallel)


@pytest.mark.parametrize("substruct_filt_cls", _get_substructural_filter_classes())
def test_substructural_filter_allowing_one_violation(substruct_filt_cls, mols_list):
    filt = substruct_filt_cls()
    filt_loose = substruct_filt_cls(allow_one_violation=True)

    mols_filtered = filt.transform(mols_list)
    mols_filtered_loose = filt_loose.transform(mols_list)

    assert len(mols_filtered) <= len(mols_filtered_loose)
    assert len(mols_filtered_loose) <= len(mols_list)


@pytest.mark.parametrize("substruct_filt_cls", _get_substructural_filter_classes())
def test_substructural_filter_return_indicators(substruct_filt_cls, mols_list):
    filt = substruct_filt_cls(return_type="indicators")
    filter_indicators = filt.transform(mols_list)

    assert_equal(len(filter_indicators), len(mols_list))
    assert isinstance(filter_indicators, np.ndarray)
    assert np.issubdtype(filter_indicators.dtype, bool)
    assert np.all(np.isin(filter_indicators, [0, 1]))


@pytest.mark.parametrize("substruct_filt_cls", _get_substructural_filter_classes())
def test_substructural_filter_transform_x_y(substruct_filt_cls, mols_list):
    labels = np.ones(len(mols_list))

    filt = substruct_filt_cls()
    mols_filtered, labels_filt = filt.transform_x_y(mols_list, labels)
    assert_equal(len(mols_filtered), len(labels_filt))


@pytest.mark.parametrize(
    "substruct_filt_cls,expected_num_conditions",
    _get_substructural_filter_classes_and_num_conditions(),
)
def test_substructural_filter_condition_names(
    substruct_filt_cls, expected_num_conditions
):
    filt = substruct_filt_cls()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (expected_num_conditions,))


@pytest.mark.parametrize(
    "substruct_filt_cls,expected_num_conditions",
    _get_substructural_filter_classes_and_num_conditions(),
)
def test_substructural_filter_return_condition_indicators(
    substruct_filt_cls, expected_num_conditions, mols_list
):
    filt = substruct_filt_cls(return_type="condition_indicators")
    condition_indicators = filt.transform(mols_list)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(mols_list), expected_num_conditions))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


@pytest.mark.parametrize(
    "substruct_filt_cls,expected_num_conditions",
    _get_substructural_filter_classes_and_num_conditions(),
)
def test_substructural_filter_return_condition_indicators_transform_x_y(
    substruct_filt_cls, expected_num_conditions, mols_list
):
    labels = np.array([1] * len(mols_list))

    filt = substruct_filt_cls(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(mols_list, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(mols_list), expected_num_conditions))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))

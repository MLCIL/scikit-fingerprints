import numpy as np
import pytest
from numpy.testing import assert_equal

from skfp.filters import TiceInsecticidesFilter


@pytest.fixture
def smiles_passing_tice_insecticides() -> list[str]:
    return [
        "O=C(CC1COc2ccccc2O1)NCCc1ccccc1",
        "Cc1cc(C)c(C)c(S(=O)(=O)Nc2ccc(OCC(=O)O)cc2)c1C",
        "O=C(Nc1cccc(Cl)c1)N1CCCC1",
    ]


@pytest.fixture
def smiles_failing_tice_insecticides() -> list[str]:
    return [
        # Lapatinib
        "CS(=O)(=O)CCNCc1ccc(o1)c2ccc3c(c2)c(ncn3)Nc4ccc(c(c4)Cl)OCc5cccc(c5)F",
        # Aliskiren
        "O=C(N)C(C)(C)CNC(=O)[C@H](C(C)C)C[C@H](O)[C@@H](N)C[C@@H](C(C)C)Cc1cc(OCCCOC)c(OC)cc1",
        # Ergotamine
        "C=12CCC=3C=C(C=C(C3[C@H](C1N=CC(=C2)Br)C4CCN(CC4)C(=O)CC5CCN(CC5)C(N)=O)Br)Cl",
    ]


@pytest.fixture
def smiles_passing_one_violation_tice_insecticides() -> list[str]:
    return [
        "CCOC(=O)C1CCCN(c2nc3ccccc3nc2C(C#N)C(=O)OC(C)COC)C1",
        "COc1ccc(C2C/C(=C3/C(=O)N=c4ccc(Cl)cc4=C3c3ccccc3)NN2S(C)(=O)=O)cc1",
        "COc1cc(C2C3=C(CC(c4ccco4)CC3=O)Nc3ccccc3N2C(C)=O)cc(OC)c1OC",
    ]


def test_mols_passing_tice_insecticides(smiles_passing_tice_insecticides):
    mol_filter = TiceInsecticidesFilter()
    smiles_filtered = mol_filter.transform(smiles_passing_tice_insecticides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_tice_insecticides))


def test_mols_failing_tice_insecticides(smiles_failing_tice_insecticides):
    mol_filter = TiceInsecticidesFilter()
    smiles_filtered = mol_filter.transform(smiles_failing_tice_insecticides)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), 0)


def test_mols_passing_with_violation_tice_insecticides(
    smiles_passing_one_violation_tice_insecticides,
):
    mol_filter = TiceInsecticidesFilter(allow_one_violation=True)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_tice_insecticides
    )
    assert_equal(len(smiles_filtered), 3)

    mol_filter = TiceInsecticidesFilter(allow_one_violation=False)
    smiles_filtered = mol_filter.transform(
        smiles_passing_one_violation_tice_insecticides
    )
    assert_equal(len(smiles_filtered), 0)


def test_tice_insecticides_return_indicators(
    smiles_passing_tice_insecticides,
    smiles_failing_tice_insecticides,
    smiles_passing_one_violation_tice_insecticides,
):
    all_smiles = (
        smiles_passing_tice_insecticides
        + smiles_failing_tice_insecticides
        + smiles_passing_one_violation_tice_insecticides
    )

    mol_filter = TiceInsecticidesFilter(return_type="indicators")
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_insecticides)
        + [False] * len(smiles_failing_tice_insecticides)
        + [False] * len(smiles_passing_one_violation_tice_insecticides),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    mol_filter = TiceInsecticidesFilter(
        allow_one_violation=True, return_type="indicators"
    )
    filter_indicators = mol_filter.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_passing_tice_insecticides)
        + [False] * len(smiles_failing_tice_insecticides)
        + [True] * len(smiles_passing_one_violation_tice_insecticides),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_tice_insecticides_parallel(smiles_list):
    mol_filter = TiceInsecticidesFilter()
    mols_filtered_sequential = mol_filter.transform(smiles_list)

    mol_filter = TiceInsecticidesFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = mol_filter.transform(smiles_list)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_tice_insecticides_transform_x_y(
    smiles_passing_tice_insecticides, smiles_failing_tice_insecticides
):
    all_smiles = smiles_passing_tice_insecticides + smiles_failing_tice_insecticides
    labels = np.array(
        [1] * len(smiles_passing_tice_insecticides)
        + [0] * len(smiles_failing_tice_insecticides)
    )

    filt = TiceInsecticidesFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_passing_tice_insecticides))
    assert np.all(labels_filt == 1)

    filt = TiceInsecticidesFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_passing_tice_insecticides))
    assert_equal(indicators, labels_filt)


def test_tice_insecticides_condition_names():
    filt = TiceInsecticidesFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (5,))


def test_tice_insecticides_return_condition_indicators(
    smiles_passing_tice_insecticides, smiles_failing_tice_insecticides
):
    all_smiles = smiles_passing_tice_insecticides + smiles_failing_tice_insecticides

    filt = TiceInsecticidesFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 5))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_tice_insecticides_return_condition_indicators_transform_x_y(
    smiles_passing_tice_insecticides, smiles_failing_tice_insecticides
):
    all_smiles = smiles_passing_tice_insecticides + smiles_failing_tice_insecticides
    labels = np.array(
        [1] * len(smiles_passing_tice_insecticides)
        + [0] * len(smiles_failing_tice_insecticides)
    )

    filt = TiceInsecticidesFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 5))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))

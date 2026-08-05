import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Mol

from skfp.filters import BeyondRo5Filter, LipinskiFilter


@pytest.fixture
def smiles_passing_ro5() -> list[str]:
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_failing_ro5() -> list[str]:
    return [
        # Telaprevir
        (
            "CCC[C@@H](C(=O)C(=O)NC1CC1)NC(=O)[C@@H]2[C@H]3CCC[C@H]"
            "3CN2C(=O)[C@H](C(C)(C)C)NC(=O)[C@H](C4CCCCC4)NC(=O)c5cnccn5"
        ),
        # Atorvastatin
        "O=C(O)C[C@H](O)C[C@H](O)CCn2c(c(c(c2c1ccc(F)cc1)c3ccccc3)C(=O)Nc4ccccc4)C(C)C",
        # Telmisartan
        "O=C(O)c1ccccc1c2ccc(cc2)Cn3c4cc(cc(c4nc3CCC)C)c5nc6ccccc6n5C",
    ]


@pytest.fixture
def smiles_beyond_ro5() -> list[str]:
    return [
        # Lapatinib
        "CS(=O)(=O)CCNCc1ccc(o1)c2ccc3c(c2)c(ncn3)Nc4ccc(c(c4)Cl)OCc5cccc(c5)F",
        # Aliskiren
        "O=C(N)C(C)(C)CNC(=O)[C@H](C(C)C)C[C@H](O)[C@@H](N)C[C@@H](C(C)C)Cc1cc(OCCCOC)c(OC)cc1",
        # Ergotamine
        "C=12CCC=3C=C(C=C(C3[C@H](C1N=CC(=C2)Br)C4CCN(CC4)C(=O)CC5CCN(CC5)C(N)=O)Br)Cl",
    ]


@pytest.fixture
def smiles_failing_beyond_ro5() -> list[str]:
    # Liraglutide peptide
    return [
        (
            "CCCCCCCCCCCCCCCC(=O)N[C@@H](CCC(=O)NCCCC[C@H](NC(=O)[C@H](C)NC(=O)[C@H](C)NC(=O)"
            "[C@H](CCC(N)=O)NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC1=CC=C(O)C=C1)"
            "NC(=O)[C@H](CO)NC(=O)[C@H](CO)NC(=O)[C@@H](NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CO)NC(=O)"
            "[C@@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@@H](NC(=O)CNC(=O)[C@H](CCC(O)=O)NC(=O)[C@H]"
            "(C)NC(=O)[C@@H](N)CC1=CN=CN1)[C@@H](C)O)[C@@H](C)O)C(C)C)C(=O)N[C@@H](CCC(O)=O)C(=O)N"
            "[C@@H](CC1=CC=CC=C1)C(=O)N[C@@H]([C@@H](C)CC)C(=O)N[C@@H](C)C(=O)N[C@@H](CC1=CNC2=CC=CC=C12)"
            "C(=O)N[C@@H](CC(C)C)C(=O)N[C@@H](C(C)C)C(=O)N[C@@H](CCCNC(N)=N)C(=O)NCC(=O)N[C@@H]"
            "(CCCNC(N)=N)C(=O)NCC(O)=O)C(O)=O"
        )
    ]


def test_mols_passing_bro5(smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5):
    filt = BeyondRo5Filter()

    smiles_filtered = filt.transform(smiles_passing_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_passing_ro5))

    smiles_filtered = filt.transform(smiles_failing_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_failing_ro5))

    smiles_filtered = filt.transform(smiles_beyond_ro5)
    assert all(isinstance(x, str) for x in smiles_filtered)
    assert_equal(len(smiles_filtered), len(smiles_beyond_ro5))


def test_mols_ro5_vs_bro5(smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5):
    all_smiles = smiles_passing_ro5 + smiles_failing_ro5 + smiles_beyond_ro5

    filt_ro5 = LipinskiFilter()
    assert_equal(len(filt_ro5.transform(smiles_passing_ro5)), 3)
    assert_equal(len(filt_ro5.transform(smiles_failing_ro5)), 0)
    assert_equal(len(filt_ro5.transform(smiles_beyond_ro5)), 0)
    assert_equal(len(filt_ro5.transform(all_smiles)), 3)

    filt_bro5 = BeyondRo5Filter()
    assert_equal(len(filt_bro5.transform(smiles_passing_ro5)), 3)
    assert_equal(len(filt_bro5.transform(smiles_failing_ro5)), 3)
    assert_equal(len(filt_bro5.transform(smiles_beyond_ro5)), 3)
    assert_equal(len(filt_bro5.transform(all_smiles)), 9)


def test_bro5_smiles_and_mol_input(smiles_list, mols_list):
    filt = BeyondRo5Filter()
    smiles_filtered = filt.transform(smiles_list)
    mols_filtered = filt.transform(mols_list)

    assert all(isinstance(x, str) for x in smiles_filtered)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert_equal(len(smiles_filtered), len(mols_filtered))


def test_bro5_parallel(smiles_list):
    filt = BeyondRo5Filter()
    mols_filtered_sequential = filt.transform(smiles_list)

    filt = BeyondRo5Filter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = filt.transform(smiles_list)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_mols_loose_bro5(mols_list):
    filt = BeyondRo5Filter()
    mols_filtered_bro5 = filt.transform(mols_list)

    filt = BeyondRo5Filter(allow_one_violation=True)
    mols_filtered_loose_bro5 = filt.transform(mols_list)

    assert len(mols_filtered_bro5) <= len(mols_filtered_loose_bro5)


def test_bro5_return_indicators(
    smiles_passing_ro5, smiles_failing_ro5, smiles_beyond_ro5
):
    all_smiles = smiles_passing_ro5 + smiles_failing_ro5 + smiles_beyond_ro5

    filt = BeyondRo5Filter(return_type="indicators")
    filter_indicators = filt.transform(all_smiles)

    assert_equal(len(filter_indicators), len(all_smiles))
    assert isinstance(filter_indicators, np.ndarray)
    assert np.issubdtype(filter_indicators.dtype, bool)
    assert np.all(np.isin(filter_indicators, [0, 1]))


def test_bro5_transform_x_y(smiles_passing_ro5, smiles_failing_beyond_ro5):
    all_smiles = smiles_passing_ro5 + smiles_failing_beyond_ro5
    labels = np.array(
        [1] * len(smiles_passing_ro5) + [0] * len(smiles_failing_beyond_ro5)
    )

    filt = BeyondRo5Filter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_passing_ro5))
    assert np.all(labels_filt == 1)

    filt = BeyondRo5Filter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_passing_ro5))
    assert np.array_equal(indicators, labels_filt)


def test_bro5_condition_names():
    filt = BeyondRo5Filter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (6,))


def test_bro5_return_condition_indicators(
    smiles_passing_ro5, smiles_failing_beyond_ro5
):
    all_smiles = smiles_passing_ro5 + smiles_failing_beyond_ro5

    filt = BeyondRo5Filter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 6))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_bro5_return_condition_indicators_transform_x_y(
    smiles_passing_ro5, smiles_failing_beyond_ro5
):
    all_smiles = smiles_passing_ro5 + smiles_failing_beyond_ro5
    labels = np.array(
        [1] * len(smiles_passing_ro5) + [0] * len(smiles_failing_beyond_ro5)
    )

    filt = BeyondRo5Filter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 6))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))

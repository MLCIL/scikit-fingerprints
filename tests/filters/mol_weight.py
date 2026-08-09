import numpy as np
import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Mol
from sklearn.utils._param_validation import InvalidParameterError

from skfp.filters import MolecularWeightFilter
from skfp.preprocessing import MolFromSmilesTransformer


@pytest.fixture
def smiles_light_mols() -> list[str]:
    # less than 200 daltons
    return [
        # paracetamol
        "CC(=O)Nc1ccc(O)cc1",
        # caffeine
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        # nicotine
        "c1ncccc1[C@@H]2CCCN2C",
    ]


@pytest.fixture
def smiles_medium_mols() -> list[str]:
    # over 500 daltons
    return [
        # Paclitaxel
        (
            "CC1=C2[C@@]([C@]([C@H]([C@@H]3[C@]4([C@H](OC4)C[C@@H]([C@]3(C(=O)[C@@H]2OC"
            "(=O)C)C)O)OC(=O)C)OC(=O)c5ccccc5)(C[C@@H]1OC(=O)[C@H](O)[C@@H](NC(=O)c6ccccc6)c7ccccc7)O)(C)C"
        ),
        # Ixabepilone
        r"Cc3nc(/C=C(\C)[C@@H]1C[C@@H]2O[C@]2(C)CCC[C@H](C)[C@H](O)[C@@H](C)C(=O)C(C)(C)[C@@H](O)CC(=O)N1)cs3",
        # Valrubicin
        (
            "FC(F)(F)C(=O)N[C@@H]5[C@H](O)[C@@H](O[C@@H](O[C@@H]4c3c(O)c2C(=O)c1c(OC)cccc"
            "1C(=O)c2c(O)c3C[C@@](O)(C(=O)COC(=O)CCCC)C4)C5)C"
        ),
    ]


@pytest.fixture
def smiles_heavy_mols() -> list[str]:
    # over 1000 daltons
    return [
        # Dactinomycin
        (
            "Cc1c2oc3c(C)ccc(C(O)=N[C@@H]4C(O)=N[C@H](C(C)C)C(=O)N5CCC[C@H]5C(=O)N(C)CC"
            "(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]4C)c3nc-2c(C(O)=N[C@@H]2C(O)=N[C@H](C(C)C)"
            "C(=O)N3CCC[C@H]3C(=O)N(C)CC(=O)N(C)[C@@H](C(C)C)C(=O)O[C@@H]2C)c(N)c1=O"
        ),
        # Plicamycin
        (
            "CO[C@H](C(=O)[C@@H](O)[C@@H](C)O)C1Cc2cc3cc(O[C@H]4C[C@@H](O[C@H]5C[C@@H](O)"
            "[C@H](O)[C@@H](C)O5)[C@@H](O)[C@@H](C)O4)c(C)c(O)c3c(O)c2C(=O)[C@H]1O[C@H]1C"
            "[C@@H](O[C@H]2C[C@@H](O[C@H]3C[C@](C)(O)[C@H](O)[C@@H](C)O3)[C@H](O)[C@@H](C)"
            "O2)[C@H](O)[C@@H](C)O1"
        ),
        # Bleomycin
        (
            "CC1=C(N=C(N=C1N)[C@H](CC(=O)N)NC[C@@H](C(=O)N)N)C(=O)N[C@@H](C(C2=CN=CN2)O[C@H]"
            "3[C@H]([C@H]([C@@H]([C@@H](O3)CO)O)O)O[C@@H]4[C@H]([C@H]([C@@H]([C@H](O4)CO)O)O"
            "C(=O)N)O)C(=O)N[C@H](C)[C@H]([C@H](C)C(=O)N[C@@H]([C@@H](C)O)C(=O)NCCC5=NC"
            "(=CS5)C6=NC(=CS6)C(=O)NCCC[S+](C)C)O"
        ),
    ]


def test_mol_weight_thresholds(
    smiles_light_mols, smiles_medium_mols, smiles_heavy_mols
):
    filt = MolecularWeightFilter()

    smiles_light_filtered = filt.transform(smiles_light_mols)
    assert all(isinstance(x, str) for x in smiles_light_filtered)
    assert_equal(len(smiles_light_filtered), len(smiles_light_mols))
    assert_equal(len(smiles_light_filtered), 3)

    smiles_medium_filtered = filt.transform(smiles_medium_mols)
    assert all(isinstance(x, str) for x in smiles_medium_filtered)
    assert_equal(len(smiles_medium_filtered), len(smiles_medium_mols))
    assert_equal(len(smiles_medium_filtered), 3)

    smiles_heavy_filtered = filt.transform(smiles_heavy_mols)
    assert_equal(len(smiles_heavy_filtered), 0)

    all_smiles = smiles_light_mols + smiles_medium_mols + smiles_heavy_mols
    all_smiles_filtered = filt.transform(all_smiles)
    assert_equal(len(all_smiles_filtered), 6)


def test_mol_weight_smiles_and_mol_input(
    smiles_light_mols, smiles_medium_mols, smiles_heavy_mols
):
    mol_from_smiles = MolFromSmilesTransformer()
    all_smiles = smiles_light_mols + smiles_medium_mols + smiles_heavy_mols
    all_mols = mol_from_smiles.transform(all_smiles)

    filt = MolecularWeightFilter()
    smiles_filtered = filt.transform(all_smiles)
    mols_filtered = filt.transform(all_mols)

    assert all(isinstance(x, str) for x in smiles_filtered)
    assert all(isinstance(x, Mol) for x in mols_filtered)
    assert_equal(len(smiles_filtered), len(mols_filtered))
    assert_equal(len(smiles_filtered), 6)


def test_mol_weight_parallel(smiles_light_mols, smiles_medium_mols, smiles_heavy_mols):
    all_smiles = smiles_light_mols + smiles_medium_mols + smiles_heavy_mols
    filt = MolecularWeightFilter()
    mols_filtered_sequential = filt.transform(all_smiles)

    filt = MolecularWeightFilter(n_jobs=-1, batch_size=1)
    mols_filtered_parallel = filt.transform(all_smiles)

    assert_equal(mols_filtered_sequential, mols_filtered_parallel)


def test_mol_weight_return_indicators(
    smiles_light_mols, smiles_medium_mols, smiles_heavy_mols
):
    all_smiles = smiles_light_mols + smiles_medium_mols + smiles_heavy_mols

    filt = MolecularWeightFilter(return_type="indicators")
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_light_mols)
        + [True] * len(smiles_medium_mols)
        + [False] * len(smiles_heavy_mols),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    filt = MolecularWeightFilter(max_weight=500, return_type="indicators")
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_light_mols)
        + [False] * len(smiles_medium_mols)
        + [False] * len(smiles_heavy_mols),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_mol_weight_min_max_thresholds(
    smiles_light_mols, smiles_medium_mols, smiles_heavy_mols
):
    all_smiles = smiles_light_mols + smiles_medium_mols + smiles_heavy_mols

    filt = MolecularWeightFilter(return_type="indicators")
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_light_mols)
        + [True] * len(smiles_medium_mols)
        + [False] * len(smiles_heavy_mols),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    filt = MolecularWeightFilter(max_weight=500, return_type="indicators")
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [True] * len(smiles_light_mols)
        + [False] * len(smiles_medium_mols)
        + [False] * len(smiles_heavy_mols),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)

    filt = MolecularWeightFilter(min_weight=500, return_type="indicators")
    filter_indicators = filt.transform(all_smiles)
    expected_indicators = np.array(
        [False] * len(smiles_light_mols)
        + [True] * len(smiles_medium_mols)
        + [False] * len(smiles_heavy_mols),
        dtype=bool,
    )
    assert_equal(filter_indicators, expected_indicators)


def test_mol_weight_wrong_min_max_thresholds(smiles_list):
    filt = MolecularWeightFilter(min_weight=1000, max_weight=100)
    with pytest.raises(InvalidParameterError) as exc_info:
        filt.transform(smiles_list)

    expected_msg = (
        "The max_weight parameter of MolecularWeightFilter must be "
        "greater or equal to min_weight, got:"
    )
    assert expected_msg in str(exc_info)


def test_mol_weight_transform_x_y(smiles_light_mols, smiles_heavy_mols):
    all_smiles = smiles_light_mols + smiles_heavy_mols
    labels = np.array([1] * len(smiles_light_mols) + [0] * len(smiles_heavy_mols))

    filt = MolecularWeightFilter()
    mols, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(len(mols), len(smiles_light_mols))
    assert np.all(labels_filt == 1)

    filt = MolecularWeightFilter(return_type="indicators")
    indicators, labels_filt = filt.transform_x_y(all_smiles, labels)
    assert_equal(np.sum(indicators), len(smiles_light_mols))
    assert_equal(indicators, labels_filt)


def test_mol_weight_condition_names():
    filt = MolecularWeightFilter()
    condition_names = filt.get_feature_names_out()

    assert isinstance(condition_names, np.ndarray)
    assert_equal(condition_names.shape, (1,))


def test_mol_weight_return_condition_indicators(smiles_light_mols, smiles_heavy_mols):
    all_smiles = smiles_light_mols + smiles_heavy_mols

    filt = MolecularWeightFilter(return_type="condition_indicators")
    condition_indicators = filt.transform(all_smiles)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 1))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))


def test_mol_weight_return_condition_indicators_transform_x_y(
    smiles_light_mols, smiles_heavy_mols
):
    all_smiles = smiles_light_mols + smiles_heavy_mols
    labels = np.array([1] * len(smiles_light_mols) + [0] * len(smiles_heavy_mols))

    filt = MolecularWeightFilter(return_type="condition_indicators")
    condition_indicators, y = filt.transform_x_y(all_smiles, labels)

    assert isinstance(condition_indicators, np.ndarray)
    assert_equal(condition_indicators.shape, (len(all_smiles), 1))
    assert np.issubdtype(condition_indicators.dtype, bool)
    assert np.all(np.isin(condition_indicators, [0, 1]))
    assert_equal(len(condition_indicators), len(y))

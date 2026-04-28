import math

import numpy as np
import pytest
from mordred import Calculator, descriptors
from mordred.error import Missing
from numpy.testing import assert_allclose
from rdkit import Chem

from skfp.fingerprints._new_mordred.cache import MordredMolCache
from skfp.fingerprints._new_mordred.calculator import compute
from skfp.fingerprints._new_mordred.descriptors import walk_count
from skfp.fingerprints._new_mordred.utils.feature_names import FEATURE_NAMES_2D

FEATURE_NAMES = [
    "MWC01",
    "MWC02",
    "MWC03",
    "MWC04",
    "MWC05",
    "MWC06",
    "MWC07",
    "MWC08",
    "MWC09",
    "MWC10",
    "TMWC10",
    "SRW02",
    "SRW03",
    "SRW04",
    "SRW05",
    "SRW06",
    "SRW07",
    "SRW08",
    "SRW09",
    "SRW10",
    "TSRW10",
]


def _expected_values(num_atoms, walk_sums, self_returning_traces):
    mwc = [0.5 * walk_sums[0]]
    mwc.extend(math.log(count + 1) for count in walk_sums[1:])

    srw = [math.log(count + 1) for count in self_returning_traces[1:]]

    return np.asarray(
        [
            *mwc,
            num_atoms + sum(mwc),
            *srw,
            num_atoms + sum(srw),
        ],
        dtype=np.float32,
    )


KNOWN_CASES = [
    ("C", _expected_values(1, [0] * 10, [0] * 10)),
    ("CC", _expected_values(2, [2] * 10, [0, 2, 0, 2, 0, 2, 0, 2, 0, 2])),
    (
        "CCC",
        _expected_values(
            3,
            [4, 6, 8, 12, 16, 24, 32, 48, 64, 96],
            [0, 4, 0, 8, 0, 16, 0, 32, 0, 64],
        ),
    ),
    (
        "c1ccccc1",
        _expected_values(
            6,
            [12, 24, 48, 96, 192, 384, 768, 1536, 3072, 6144],
            [0, 12, 0, 36, 0, 132, 0, 516, 0, 2052],
        ),
    ),
    ("C.C", _expected_values(2, [0] * 10, [0] * 10)),
    ("CC.C", _expected_values(3, [2] * 10, [0, 2, 0, 2, 0, 2, 0, 2, 0, 2])),
]

SMILES = ["C", "CC", "CCC", "CC.C", "C=C", "c1ccccc1", "C1CCCCC1"]


@pytest.fixture(scope="module")
def mordred_walk_count_calc():
    return Calculator(descriptors.WalkCount, ignore_3D=True)


def _mordred_values(mol, calc):
    values = dict(
        zip((str(desc) for desc in calc.descriptors), calc(mol), strict=False)
    )
    return np.asarray(
        [
            np.nan if isinstance(values[name], Missing) else values[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float32,
    )


def test_walk_count_feature_names_and_order():
    assert walk_count.FEATURE_NAMES == FEATURE_NAMES
    assert len(walk_count.FEATURE_NAMES) == 21

    names = FEATURE_NAMES_2D
    idxs = [names.index(name) for name in FEATURE_NAMES]
    assert idxs == list(range(names.index("MWC01"), names.index("TSRW10") + 1))
    assert names.index("VAdjMat") < names.index("MWC01") < names.index("MW")


@pytest.mark.parametrize(("smiles", "expected"), KNOWN_CASES)
def test_walk_count_known_values(smiles, expected):
    cache = MordredMolCache.from_mol(Chem.MolFromSmiles(smiles), use_3D=False)

    values, feature_names = walk_count.calc(cache)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_walk_count_totals_sum_descriptor_values_not_raw_counts():
    cache = MordredMolCache.from_mol(Chem.MolFromSmiles("CCC"), use_3D=False)

    values, feature_names = walk_count.calc(cache)
    observed = dict(zip(feature_names, values, strict=False))

    raw_total = 3 + sum([4, 6, 8, 12, 16, 24, 32, 48, 64, 96])
    assert observed["TMWC10"] == pytest.approx(
        3 + sum(observed[f"MWC{order:02d}"] for order in range(1, 11))
    )
    assert observed["TMWC10"] != pytest.approx(raw_total)
    assert observed["TSRW10"] == pytest.approx(
        3 + sum(observed[f"SRW{order:02d}"] for order in range(2, 11))
    )


@pytest.mark.parametrize("smiles", SMILES)
def test_walk_count_matches_mordred(smiles, mordred_walk_count_calc):
    mol = Chem.MolFromSmiles(smiles)
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = walk_count.calc(cache)
    expected = _mordred_values(mol, mordred_walk_count_calc)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


def test_walk_count_ignores_explicit_hydrogens_like_mordred(mordred_walk_count_calc):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    cache = MordredMolCache.from_mol(mol, use_3D=False)

    values, feature_names = walk_count.calc(cache)
    expected = _mordred_values(mol, mordred_walk_count_calc)

    assert feature_names == FEATURE_NAMES
    assert_allclose(values, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("smiles", SMILES)
def test_calculator_fills_walk_count_columns(smiles, mordred_walk_count_calc):
    mol = Chem.MolFromSmiles(smiles)

    observed = compute(mol, use_3D=False)
    idxs = [FEATURE_NAMES_2D.index(name) for name in FEATURE_NAMES]
    expected = _mordred_values(mol, mordred_walk_count_calc)

    assert_allclose(observed[idxs], expected, rtol=1e-6, atol=1e-6)

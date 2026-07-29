import pytest
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import ring_count
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

FEATURE_NAMES = ring_count.FEATURE_NAMES
GENERAL_RING_FEATURE_NAMES = [
    "nRing",
    "nHRing",
    "naRing",
    "naHRing",
    "nARing",
    "nAHRing",
]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("C1CC1", {"nRing": 1, "nARing": 1, "n3Ring": 1}),
        (
            "c1ccncc1",
            {
                "nRing": 1,
                "nHRing": 1,
                "naRing": 1,
                "naHRing": 1,
                "n6Ring": 1,
                "n6HRing": 1,
                "n6aRing": 1,
                "n6aHRing": 1,
            },
        ),
        (
            "C1CCC2CCCCC2C1",
            {
                "nRing": 2,
                "nARing": 2,
                "n6Ring": 2,
                "n6ARing": 2,
                "nFRing": 1,
                "n10FRing": 1,
                "nFARing": 1,
                "n10FARing": 1,
            },
        ),
        (
            "c1ccc2ccccc2c1",
            {
                "nRing": 2,
                "naRing": 2,
                "n6Ring": 2,
                "n6aRing": 2,
                "nFRing": 1,
                "n10FRing": 1,
                "nFaRing": 1,
                "n10FaRing": 1,
            },
        ),
        ("C1CCC2(CC1)CCCC2", {"nRing": 2, "n5Ring": 1, "n6Ring": 1}),
        ("C1CCCCCCCCCCCC1", {"nRing": 1, "nARing": 1, "nG12Ring": 1}),
    ],
)
def test_ring_count_values(smiles, expected):
    mol = Chem.MolFromSmiles(smiles)

    rings = ring_count.RingSets(mol, AtomicProperties(mol))
    values = ring_count.calc(rings)
    values_by_name = dict(zip(FEATURE_NAMES, values, strict=True))
    for name, expected_value in expected.items():
        assert values_by_name[name] == expected_value


def test_ring_count_includes_general_ring_descriptors():
    assert ring_count.FEATURE_NAMES[: len(GENERAL_RING_FEATURE_NAMES)] == (
        GENERAL_RING_FEATURE_NAMES
    )

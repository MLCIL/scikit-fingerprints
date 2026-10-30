import pytest
from rdkit import Chem

from skfp.fingerprints._new_mordred.descriptors import ring_count
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

FEATURE_NAMES = ring_count.FEATURE_NAMES
GENERAL_RING_FEATURE_NAMES = [
    "nRing",
    "number of heterocycles",
    "number of aromatic rings",
    "number of aromatic heterocycles",
    "number of aliphatic rings",
    "number of aliphatic heterocycles",
]


@pytest.mark.parametrize(
    "smiles, expected",
    [
        (
            "C1CC1",
            {
                "nRing": 1,
                "number of aliphatic rings": 1,
                "number of 3-membered rings": 1,
            },
        ),
        (
            "c1ccncc1",
            {
                "nRing": 1,
                "number of heterocycles": 1,
                "number of aromatic rings": 1,
                "number of aromatic heterocycles": 1,
                "number of 6-membered rings": 1,
                "number of 6-membered heterocycles": 1,
                "number of 6-membered aromatic rings": 1,
                "number of 6-membered aromatic heterocycles": 1,
            },
        ),
        (
            "C1CCC2CCCCC2C1",
            {
                "nRing": 2,
                "number of aliphatic rings": 2,
                "number of 6-membered rings": 2,
                "number of 6-membered aliphatic rings": 2,
                "number of fused ring systems": 1,
                "number of 10-atom fused ring systems": 1,
                "number of fused aliphatic ring systems": 1,
                "number of 10-atom fused aliphatic ring systems": 1,
            },
        ),
        (
            "c1ccc2ccccc2c1",
            {
                "nRing": 2,
                "number of aromatic rings": 2,
                "number of 6-membered rings": 2,
                "number of 6-membered aromatic rings": 2,
                "number of fused ring systems": 1,
                "number of 10-atom fused ring systems": 1,
                "number of fused aromatic ring systems": 1,
                "number of 10-atom fused aromatic ring systems": 1,
            },
        ),
        (
            "C1CCC2(CC1)CCCC2",
            {
                "nRing": 2,
                "number of 5-membered rings": 1,
                "number of 6-membered rings": 1,
            },
        ),
        (
            "C1CCCCCCCCCCCC1",
            {
                "nRing": 1,
                "number of aliphatic rings": 1,
                "number of 12+-membered rings": 1,
            },
        ),
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

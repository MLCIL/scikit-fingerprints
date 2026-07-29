import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.bcut import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""


def _calc(mol: Mol) -> tuple[np.ndarray, list[str]]:
    return calc(AtomicProperties(mol), len(GetMolFrags(mol)))


@pytest.mark.parametrize(
    "name, expected_smallest, expected_largest",
    [
        ("Hexane", 11.89, 12.11007471),
        ("Benzene", 11.85, 12.1500544),
        ("Caffeine", 11.99347887, 15.99592273),
        ("Cyanidin", 11.85, 15.99993732),
        ("Lycopene", 11.89, 12.11102666),
        ("Epicatechin", 11.85, 15.99993732),
        ("Limonene", 11.9, 12.10017173),
        ("Allicin", 11.79, 31.97307173),
        ("Glutathione", 11.99846101, 31.97267783),
        ("Digoxin", 11.89, 16.00805947),
        ("Capsaicin", 11.89, 15.99692975),
        ("EllagicAcid", 11.85, 16.00194285),
        ("Astaxanthin", 11.89, 15.99795509),
    ],
)
def test_bcut_mass_values(name, expected_smallest, expected_largest, mordred_test_mols):
    mol = mordred_test_mols[name]

    values = _calc(mol)
    values = dict(zip(FEATURE_NAMES, values, strict=True))
    actual_smallest = values["BCUT_mass_smallest_eigval"]
    actual_largest = values["BCUT_mass_largest_eigval"]

    # Mordred tests also use such a large tolerance
    assert_allclose(actual_smallest, expected_smallest, atol=1)
    assert_allclose(actual_largest, expected_largest, atol=1)


def test_disconnected_mol_all_nan():
    mol = MolFromSmiles("[Na].[Cl]")
    values = _calc(mol)
    assert_allclose(values, np.nan)


# skfp property name -> the abbreviation Mordred uses for the same atomic property
_MORDRED_PROPERTY_ABBREVIATIONS = {
    "gasteiger_charge": "c",
    "valence_electrons": "dv",
    "sigma_electrons": "d",
    "intrinsic_state": "s",
    "atomic_number": "Z",
    "mass": "m",
    "van_der_Waals_volume": "v",
    "Sanderson_electronegativity": "se",
    "Pauling_electronegativity": "pe",
    "Allred_Rochow_electronegativity": "are",
    "polarizability": "p",
    "ionization_potential": "i",
}

# Mordred's suffix for the highest and the lowest eigenvalue
_MORDRED_EIGENVALUE_SUFFIXES = {"largest": "1h", "smallest": "1l"}


def _mordred_name(feature_name: str) -> str:
    """Translate a BCUT feature name into the matching Mordred descriptor name."""
    prop, kind = (
        feature_name.removeprefix("BCUT_")
        .removesuffix("_eigval")
        .rsplit("_", maxsplit=1)
    )
    abbreviation = _MORDRED_PROPERTY_ABBREVIATIONS[prop]
    return f"BCUT{abbreviation}-{_MORDRED_EIGENVALUE_SUFFIXES[kind]}"


@pytest.mark.parametrize(
    "smiles",
    [
        "CC",
        "CCO",
        "c1ccccc1",
        "CC(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "O=S(=O)(O)O",
        "N[C@@H](CCC(=O)O)C(=O)O",
    ],
)
def test_bcut_matches_mordred_by_feature_name(smiles):
    """
    Every BCUT descriptor must match Mordred's value for the same atomic property
    and eigenvalue, looked up by name so that the result does not depend on the
    order in which the descriptors happen to be computed.
    """
    mordred = pytest.importorskip("mordred")

    mol = MolFromSmiles(smiles)
    values = _calc(mol)
    actual = dict(zip(FEATURE_NAMES, values, strict=True))

    calculator = mordred.Calculator(mordred.descriptors.BCUT, ignore_3D=True)
    expected = {
        str(descriptor): value
        for descriptor, value in zip(
            calculator.descriptors, calculator(mol), strict=True
        )
    }

    for feature_name, value in actual.items():
        assert_allclose(
            value,
            float(expected[_mordred_name(feature_name)]),
            atol=1e-4,
            err_msg=feature_name,
        )

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags, Mol, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.molecular_id import FEATURE_NAMES, calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values for MID and AMID come from mordred-community's own reference
data (mordred/tests/references/MolecularId.yaml) and are stored at
./references/molecular_id.json as expected[molecule][descriptor]. The
element-filtered features are not covered there, so their values below were
computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "molecular_id.json") as f:
    _REFERENCE = json.load(f)


def _calc(mol: Mol) -> dict[str, float]:
    values = calc(AtomicProperties(mol), len(GetMolFrags(mol)))
    return dict(zip(FEATURE_NAMES, values, strict=True))


def _calc_from_smiles(smiles: str) -> dict[str, float]:
    return _calc(preprocess_mol(MolFromSmiles(smiles)))


@pytest.mark.parametrize("molecule", list(_REFERENCE))
def test_molecular_id_reference_values(molecule, mordred_test_mols):
    values = _calc(mordred_test_mols[molecule])

    expected = _REFERENCE[molecule]
    assert_allclose(
        [values[name] for name in expected],
        list(expected.values()),
        rtol=1e-5,
    )


@pytest.mark.parametrize(
    "smiles, feature, expected",
    [
        # a lone heavy atom has no paths, so its atomic id stays at 1
        ("N", "MID", 1.0),
        ("N", "MID nitrogens", 1.0),
        ("CCO", "MID carbons", 3.310660),
        ("CCO", "MID oxygens", 1.603553),
        ("CCO", "MID heteroatoms", 1.603553),
        ("CCO", "MID nitrogens", 0.0),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "MID nitrogens", 8.359750),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "MID oxygens", 3.521046),
        ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "MID heteroatoms", 11.880796),
        ("CS(=O)(=O)N", "MID nitrogens", 1.625),
        ("CS(=O)(=O)N", "MID oxygens", 3.25),
        # every halogen is also a heteroatom, and nothing here is carbon but one atom
        ("FC(F)(Cl)Br", "MID halogens", 6.5),
        ("FC(F)(Cl)Br", "MID heteroatoms", 6.5),
        ("FC(F)(Cl)Br", "MID carbons", 2.0),
        ("Clc1ccccc1", "MID halogens", 1.745348),
        # hydrogens that RemoveHs keeps are not heteroatoms, unlike every other
        # non-carbon atom
        ("[H][H]", "MID", 3.0),
        ("[H][H]", "MID heteroatoms", 0.0),
        ("[2H]C([2H])([2H])O", "MID heteroatoms", 1.625),
    ],
)
def test_element_filtered_ids(smiles, feature, expected):
    values = _calc_from_smiles(smiles)

    assert_allclose(values[feature], expected, rtol=1e-5, atol=1e-6)


def test_averaged_ids_divide_by_every_heavy_atom():
    # AMID carbons averages the carbon-only sum over all three heavy atoms, not over
    # the two carbons
    values = _calc_from_smiles("CCO")

    assert_allclose(values["AMID carbons"], values["MID carbons"] / 3, rtol=1e-6)
    assert_allclose(values["AMID oxygens"], values["MID oxygens"] / 3, rtol=1e-6)


def test_disconnected_molecule_is_nan():
    # atomic ids are undefined without a connected graph; mordred reports
    # "multiple fragments" for these
    values = _calc_from_smiles("[Na+].[Cl-]")

    assert len(values) == len(FEATURE_NAMES)
    assert all(np.isnan(value) for value in values.values())

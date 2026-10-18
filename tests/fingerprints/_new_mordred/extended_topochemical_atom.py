import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags, MolFromSmiles
from rdkit.Chem.rdchem import Bond

from skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.descriptors.ring_count import RingSets
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import (
    bonds_apply_func,
    preprocess_mol,
)

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were generated with mordred-community and are stored at
./references/eta.json as expected[molecule][feature_name], with null for NaN.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

with open(Path(__file__).parent / "references" / "eta.json") as file:
    _REFERENCE = json.load(file)


def _compute(mol):
    # inputs are built the same way calculator.compute builds them
    n_frags = len(GetMolFrags(mol))

    mol_regular = preprocess_mol(mol)
    distance_matrix_regular = DistanceMatrix.from_mol(mol_regular)
    props_regular = AtomicProperties(mol_regular)
    rings_regular = RingSets(mol_regular, props_regular)

    mol_hydrogens = preprocess_mol(mol, explicit_hydrogens=True)
    props_hydrogens = AtomicProperties(mol_hydrogens)

    mol_kekulized = preprocess_mol(mol_regular, kekulize=True, sanitize=False)
    kekulized_bond_types = bonds_apply_func(Bond.GetBondType, mol_kekulized, np.intp)

    mol_kekulized_hydrogens = preprocess_mol(
        mol_hydrogens, kekulize=True, sanitize=False
    )
    kekulized_bond_types_hydrogens = bonds_apply_func(
        Bond.GetBondType, mol_kekulized_hydrogens, np.intp
    )

    values = calc(
        kekulized_bond_types,
        kekulized_bond_types_hydrogens,
        props_regular,
        props_hydrogens,
        distance_matrix_regular,
        rings_regular,
        n_frags,
    )
    return dict(zip(FEATURE_NAMES, values, strict=True))


@pytest.mark.parametrize("molecule", list(_REFERENCE["expected"]))
def test_eta_reference_values(molecule, mordred_test_mols):
    mol = mordred_test_mols[molecule]
    computed = _compute(mol)

    expected = _REFERENCE["expected"][molecule]
    vals_mordred = np.array(
        [np.nan if expected[f] is None else expected[f] for f in computed],
        dtype=float,
    )
    vals_skfp = np.array([computed[f] for f in computed], dtype=float)

    assert_allclose(vals_skfp, vals_mordred, atol=1e-3, equal_nan=True)


def test_disconnected_mol_all_nan():
    mol = MolFromSmiles("[Na].[Cl]")
    computed = _compute(mol)
    assert_allclose(list(computed.values()), np.nan)


# the saturated reference variant fills free valences with hydrogens the way RDKit
# does, which for charged atoms follows the isoelectronic element; expected values
# are from mordred-community
@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("[CH2+]C", 0.4142857142857142),  # carbocation, valence 3 like boron
        ("C[C+2]C", 0.4333333333333332),
        ("C[N+2]C", 0.45999999999999985),
        ("[Si-](C)(C)(C)(C)C", 0.37857142857142845),  # valence 5 like phosphorus
        ("[P-](F)(F)(F)(F)(F)F", 1.6265306122448984),  # valence 6 like sulfur
        ("[PbH2+2]", -2.7),  # a lone cation takes no hydrogens
        ("N->[Pt](Cl)Cl", 0.9159183673469389),  # dative bond
        ("O->[Fe](Cl)(Cl)Cl", 1.124829931972789),
        ("C[Mn]1[H-][Mn]1C", 0.5685714285714285),  # bridging hydride
        # hydralazine, whose fused ring kekulizes one way with the hydrogens
        # explicit and another without them, changing how many hydrogens the
        # saturated variant takes
        ("NNc1nncc2ccccc12", 0.5285714285714286),
    ],
)
def test_eta_epsilon_4_saturated_variant_hydrogens(smiles, expected):
    computed = _compute(MolFromSmiles(smiles))
    assert_allclose(computed["ETA_epsilon_4"], expected, rtol=1e-5)


# Mordred builds the alkane reference of ETA_epsilon_3 on the hydrogen-explicit
# molecule and gives up when an atom there has more than four neighbors, hydrogens
# included, e.g. a phosphorus with four carbons and a hydrogen; the hydrogen-free
# alkane reference of the other ETA descriptors still exists. Expected values are
# from mordred-community.
@pytest.mark.parametrize(
    "smiles",
    [
        "O=C1O[IH2](C#Cc2ccccc2)c2ccccc21",
        (
            "C1=C(c2cnc3ccccc3n2)[SH+][Pd-2]2([SH+]1)[PH](c1ccccc1)(c1ccccc1)CC[PH]2"
            "(c1ccccc1)c1ccccc1"
        ),
        "N#CCC[PH](c1ccccc1)(c1ccccc1)c1ccccc1",
        "Cc1ccccc1[PH](Cc1ccccc1)(c1ccccc1)c1ccccc1",
        "C=CC[PH](CC=C)(c1ccccc1)c1ccccc1",
    ],
)
def test_eta_epsilon_3_undefined_for_more_than_four_neighbors_with_hydrogens(smiles):
    computed = _compute(MolFromSmiles(smiles))
    values = [
        computed[f] for f in ("ETA_epsilon_3", "ETA_dEpsilon_A", "ETA_dEpsilon_C")
    ]
    assert_allclose(values, np.nan)


# without any heavy atom the alkane reference is empty, as in Mordred: its sums are
# zero, while averaging over its atoms is undefined, and so are the shape and the
# saturated variant, which divide by zero; expected values are from mordred-community
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("smiles", ["[H][H]", "[2H][2H]"])
def test_eta_reference_without_heavy_atoms_is_empty(smiles):
    computed = _compute(MolFromSmiles(smiles))
    expected = {
        "ETA_eta_R": 0.0,
        "AETA_eta_R": np.nan,
        "ETA_eta_RL": 0.0,
        "AETA_eta_RL": np.nan,
        "ETA_eta_B": 1.0,
        "AETA_eta_B": 0.5,
        "ETA_eta_BR": 1.0,
        "AETA_eta_BR": 0.5,
        "ETA_dAlpha_A": 0.0,
        "ETA_dAlpha_B": 0.0,
        "ETA_epsilon_3": np.nan,
        "ETA_shape_p": np.nan,
        "ETA_shape_y": np.nan,
        "ETA_shape_x": np.nan,
        "ETA_epsilon_4": np.nan,
    }
    assert_allclose([computed[name] for name in expected], list(expected.values()))

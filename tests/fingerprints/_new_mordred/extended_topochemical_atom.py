import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from rdkit.Chem import GetMolFrags, GetSymmSSSR, MolFromSmiles

from skfp.fingerprints._new_mordred.descriptors.extended_topochemical_atom import (
    FEATURE_NAMES,
    calc,
)
from skfp.fingerprints._new_mordred.utils.graph_matrix import DistanceMatrix
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol

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
    mol_kekulized = preprocess_mol(mol, kekulize=True)
    distance_matrix = DistanceMatrix(mol_kekulized)
    mol_kekulized_hydrogens = preprocess_mol(
        mol, kekulize=True, explicit_hydrogens=True
    )
    ring_count = len(GetSymmSSSR(mol_kekulized))
    n_frags = len(GetMolFrags(mol))

    values = calc(
        mol_kekulized,
        distance_matrix,
        mol_kekulized_hydrogens,
        ring_count,
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

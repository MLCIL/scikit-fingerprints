import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfp.fingerprints._new_mordred.descriptors.kappa_shape_index import calc
from skfp.fingerprints._new_mordred.utils.atomic_properties import AtomicProperties
from skfp.fingerprints._new_mordred.utils.mol_preprocess import preprocess_mol
from skfp.fingerprints._new_mordred.utils.subgraphs import Subgraphs

"""
This code has been adapted from the BSD-licensed mordred-community library.
https://github.com/JacksonBurns/mordred-community

Reference values were computed with mordred-community.

See skfp/fingerprints/data/mordred-community_bsd_license.txt for the license text.
"""

NAN = float("nan")


@pytest.mark.parametrize(
    "name, expected",
    [
        # too few atoms to span a path of the given order
        ("Ammonia", [NAN, NAN, NAN]),
        ("SodiumChloride", [NAN, NAN, NAN]),  # two ions, so no bonds at all
        ("Acetylene", [2.0, NAN, NAN]),
        ("Acetonitrile", [3.0, 2.0, NAN]),
        # chains score highest, branching pulls the indices down
        ("Hexane", [6.0, 5.0, 5.33333]),
        ("Triethoxyphosphine", [10.0, 7.11111, 5.53086]),
        ("PentachloroLambda5Phosphane", [6.0, 0.8, NAN]),
        # rings
        ("Benzene", [4.16667, 2.22222, 1.33333]),
        ("Thiophene", [3.2, 1.44, 0.64]),
        ("MethylCyclopropane", [2.25, 0.48, 1.0]),
        # drug-like and fused ring systems
        ("Caffeine", [10.51556, 3.53875, 1.45455]),
        ("Capsaicin", [20.04545, 11.52263, 9.03686]),
        ("EllagicAcid", [15.5232, 5.25, 2.11111]),
        ("Histidine", [9.09091, 4.13265, 2.84444]),
        # large molecules
        ("Astaxanthin", [40.1758, 18.51855, 12.8576]),
        ("Lycopene", [40.0, 26.61437, 28.89562]),
        ("Digoxin", [41.72216, 15.79404, 7.3445]),
    ],
)
def test_kappa_shape_index_values(name, expected, mordred_test_mols):
    mol_regular = preprocess_mol(mordred_test_mols[name])

    props = AtomicProperties.from_mol(mol_regular)
    values = calc(props, Subgraphs(props))

    assert_allclose(
        values, np.asarray(expected, dtype=np.float32), rtol=1e-5, equal_nan=True
    )

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from rdkit.Chem import MolFromSmiles
from scipy.sparse import csr_array

from skfp.fingerprints import MAGPIEFingerprint
from skfp.fingerprints.magpie import _ELEMENTAL_PROPERTIES, _load_elemental_data


def test_magpie_fingerprint(smiles_list):
    magpie_fp = MAGPIEFingerprint(n_jobs=-1)
    X = magpie_fp.transform(smiles_list)

    assert isinstance(X, np.ndarray)
    assert np.issubdtype(X.dtype, np.floating)
    assert_equal(X.shape, (len(smiles_list), 145))
    assert not np.any(np.isnan(X))


def test_magpie_sparse_fingerprint(smiles_list):
    magpie_fp = MAGPIEFingerprint(sparse=True, n_jobs=-1)
    X = magpie_fp.transform(smiles_list)

    assert isinstance(X, csr_array)
    assert np.issubdtype(X.dtype, np.floating)
    assert_equal(X.shape, (len(smiles_list), 145))


def test_magpie_parallel_equals_sequential(smiles_list):
    X_seq = MAGPIEFingerprint().transform(smiles_list)
    X_par = MAGPIEFingerprint(n_jobs=-1, batch_size=1).transform(smiles_list)

    assert_allclose(X_seq, X_par)


def test_magpie_feature_names():
    magpie_fp = MAGPIEFingerprint()
    feature_names = magpie_fp.get_feature_names_out()

    assert_equal(len(feature_names), magpie_fp.n_features_out)
    assert_equal(len(feature_names), len(set(feature_names)))

    assert_equal(feature_names[0], "number of elements")
    assert_equal(feature_names[1], "L2 norm of element fractions")
    assert_equal(feature_names[6], "mean atomic number")
    assert_equal(feature_names[138], "fraction of s valence electrons")
    assert_equal(feature_names[-1], "mean ionic character")


def test_magpie_depends_only_on_formula():
    # ethanol and dimethyl ether are isomers, so MAGPIE cannot distinguish them
    X = MAGPIEFingerprint().transform(["CCO", "COC"])
    assert_allclose(X[0], X[1])

    # the same formula, but written with explicit hydrogens
    X = MAGPIEFingerprint().transform(["CCO", "[H]C([H])([H])C([H])([H])O[H]"])
    assert_allclose(X[0], X[1])


def test_magpie_composition():
    atomic_nums, fractions = MAGPIEFingerprint()._get_composition(MolFromSmiles("CCO"))
    assert_array_equal(atomic_nums, [1, 6, 8])
    assert_allclose(fractions, [6 / 9, 2 / 9, 1 / 9])

    # hydrogen-free molecule must not get a zero-fraction hydrogen entry
    atomic_nums, fractions = MAGPIEFingerprint()._get_composition(
        MolFromSmiles("[Na+].[Cl-]")
    )
    assert_array_equal(atomic_nums, [11, 17])
    assert_allclose(fractions, [0.5, 0.5])


def test_magpie_water_values():
    X = MAGPIEFingerprint().transform(["O"])[0]
    feature_names = list(MAGPIEFingerprint().get_feature_names_out())

    def feature(name: str) -> float:
        return X[feature_names.index(name)]

    # H2O: 2/3 hydrogen, 1/3 oxygen
    assert_allclose(feature("number of elements"), 2)
    assert_allclose(feature("L2 norm of element fractions"), np.sqrt(5) / 3)
    assert_allclose(feature("mean atomic number"), 2 / 3 * 1 + 1 / 3 * 8)
    assert_allclose(feature("min atomic number"), 1)
    assert_allclose(feature("max atomic number"), 8)
    assert_allclose(feature("range atomic number"), 7)
    # mode is hydrogen, which is the most prevalent element
    assert_allclose(feature("mode atomic number"), 1)
    # mean atomic number is 10/3, so the deviations are 7/3 and 14/3
    assert_allclose(
        feature("mean abs deviation atomic number"), 2 / 3 * 7 / 3 + 1 / 3 * 14 / 3
    )

    # composition-weighted mean valence electrons: s = 2/3 * 1 + 1/3 * 2 = 4/3,
    # p = 2/3 * 0 + 1/3 * 4 = 4/3, d = f = 0
    assert_allclose(feature("fraction of s valence electrons"), 0.5)
    assert_allclose(feature("fraction of p valence electrons"), 0.5)
    assert_allclose(feature("fraction of d valence electrons"), 0)
    assert_allclose(feature("fraction of f valence electrons"), 0)

    # H(+1) and O(-2) can form a charge-neutral compound
    assert_allclose(feature("can form ionic compound"), 1)
    ionic_char = 1 - np.exp(-0.25 * (3.44 - 2.2) ** 2)
    assert_allclose(feature("max ionic character"), ionic_char)
    assert_allclose(feature("mean ionic character"), 2 * (2 / 3) * (1 / 3) * ionic_char)


def test_magpie_mode_ties_are_averaged():
    # benzene is C6H6, so both elements are equally prevalent, and the original
    # MAGPIE implementation averages their property values
    X = MAGPIEFingerprint().transform(["c1ccccc1"])[0]
    feature_names = list(MAGPIEFingerprint().get_feature_names_out())
    assert_allclose(X[feature_names.index("mode atomic number")], (1 + 6) / 2)


def test_magpie_can_form_ionic():
    smiles = [
        "[Na+].[Cl-]",  # NaCl: Na(+1) and Cl(-1) balance out
        "O",  # H2O: 2 * H(+1) and O(-2) balance out
        "C",  # CH4: C(-4) and 4 * H(+1) balance out
        "[H][H]",  # single element, cannot be ionic
        "[C]",  # single element, cannot be ionic
        "CC",  # C2H6: no combination of C(-4, 2, 4) and H(-1, 1) balances out
        "c1ccccc1",  # C6H6: same as above, with equal element fractions
        "[He]",  # helium has no known oxidation states
    ]
    X = MAGPIEFingerprint().transform(smiles)
    feature_names = list(MAGPIEFingerprint().get_feature_names_out())
    idx = feature_names.index("can form ionic compound")

    assert_allclose(X[:, idx], [1, 1, 1, 0, 0, 0, 0, 0])


def test_magpie_missing_property_gives_nan():
    # electronegativity and melting temperature of helium are undefined
    X = MAGPIEFingerprint().transform(["[He]"])[0]
    feature_names = list(MAGPIEFingerprint().get_feature_names_out())

    for name in feature_names:
        value = X[feature_names.index(name)]
        if (
            "electronegativity" in name
            or "melting temperature" in name
            or "ionic char" in name
        ):
            assert np.isnan(value), name
        else:
            assert not np.isnan(value), name


def test_magpie_unsupported_element():
    # oganesson (Z = 118) is outside MAGPIE lookup tables
    with pytest.raises(ValueError, match="atomic numbers 1-112"):
        MAGPIEFingerprint().transform(["[Og]"])


def test_magpie_empty_molecule():
    with pytest.raises(ValueError, match="empty molecule"):
        MAGPIEFingerprint().transform([""])


def test_magpie_elemental_data():
    properties, oxidation_states = _load_elemental_data()

    # elements from hydrogen to copernicium, with 22 properties each
    assert_equal(properties.shape, (112, 22))
    assert_equal(len(oxidation_states), 112)

    # hydrogen is the first element
    assert_allclose(properties[0, _ELEMENTAL_PROPERTIES.index("atomic number")], 1)
    assert_allclose(
        properties[0, _ELEMENTAL_PROPERTIES.index("electronegativity")], 2.2
    )
    assert_equal(oxidation_states[0], (-1, 1))

    # helium has no known oxidation states and undefined electronegativity
    assert np.isnan(properties[1, _ELEMENTAL_PROPERTIES.index("electronegativity")])
    assert_equal(oxidation_states[1], ())

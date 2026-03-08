import pytest
from rdkit.Chem import MolFromSmiles

from skfp.fingerprints import AtomPairFingerprint
from skfp.utils.validators import (
    ensure_mols,
    ensure_smiles,
    require_atoms,
    require_mols,
    require_mols_with_conf_ids,
    require_strings,
)


def test_ensure_mols(mols_list):
    ensure_mols(mols_list)
    with pytest.raises(TypeError) as exc_info:
        ensure_mols(mols_list + [1])

    assert "Passed values must be RDKit Mol objects" in str(exc_info)


def test_ensure_mols_wrong_smiles():
    smiles_list = ["O", "O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3"]
    with pytest.raises(TypeError) as exc_info:
        ensure_mols(smiles_list)

    assert "Could not parse" in str(exc_info)
    assert "at index 1 as molecule" in str(exc_info)


def test_ensure_mols_valid_inchi():
    inchi_list = ["InChI=1S/H2O/h1H2", "InChI=1S/CH4/h1H4"]
    mols = ensure_mols(inchi_list)
    assert all(m is not None for m in mols)
    assert len(mols) == 2
    from rdkit.Chem import MolToSmiles

    smiles = [MolToSmiles(m) for m in mols]
    assert "O" in smiles
    assert "C" in smiles


def test_ensure_mols_invalid_inchi():
    inchi_list = ["InChI=1S/H2O/h1H2", "InChI=1S/invalid"]
    with pytest.raises(TypeError) as exc_info:
        ensure_mols(inchi_list)

    assert "Could not parse" in str(exc_info)
    assert "at index 1 as molecule" in str(exc_info)


def test_ensure_mols_in_fingerprint():
    smiles_list = ["O", "O=N([O-])C1=C(CN=C1NCCSCc2ncccc2)Cc3ccccc3"]
    fp = AtomPairFingerprint()
    with pytest.raises(TypeError) as exc_info:
        fp.transform(smiles_list)

    assert "Could not parse" in str(exc_info)
    assert "at index 1 as molecule" in str(exc_info)


def test_ensure_smiles(smiles_list):
    ensure_smiles(smiles_list)
    with pytest.raises(TypeError, match="Passed values must be SMILES strings"):
        ensure_smiles(smiles_list + [1])


def test_require_mols(mols_list):
    require_mols(mols_list)
    with pytest.raises(TypeError, match="Passed values must be RDKit Mol objects"):
        require_mols(mols_list + [1])


def test_require_strings(smiles_list):
    require_strings(smiles_list)
    with pytest.raises(TypeError, match="Passed values must be strings"):
        require_strings(smiles_list + [1])


def test_require_mols_with_conf_ids(mols_conformers_list, mols_list):
    require_mols_with_conf_ids(mols_conformers_list)
    with pytest.raises(TypeError) as exc_info:
        require_mols_with_conf_ids(mols_conformers_list + mols_list)
    assert "each must have conf_id property set" in str(exc_info)


@pytest.mark.parametrize(
    "min_atoms, only_explicit, expected_message",
    [
        (1, True, None),  # Basic case: 1 atom required, water has 1 explicit atom
        (
            2,
            True,
            "The molecule must have at least 2 atom(s)",
        ),  # Failure case: water has only 1 explicit atom
        (
            2,
            False,
            None,
        ),  # Special case: water has 3 atoms when counting implicit hydrogens
    ],
)
def test_require_atoms_decorator(min_atoms, only_explicit, expected_message):
    """Test the require_atoms decorator with different configurations."""

    # Create a decorated test function
    @require_atoms(min_atoms=min_atoms, only_explicit=only_explicit)
    def test_function(molecule):
        return True

    water_molecule = MolFromSmiles("O")

    if expected_message:
        with pytest.raises(ValueError) as exc_info:
            test_function(water_molecule)
        assert expected_message in str(exc_info.value)
    else:
        assert test_function(water_molecule) is True

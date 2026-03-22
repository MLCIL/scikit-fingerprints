import os

import pytest
from numpy.testing import assert_equal
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSDFTransformer, MolToSDFTransformer
from skfp.preprocessing.input_output import sdf as sdf_module


@pytest.fixture
def sdf_in_file_path():
    # L-alanine
    # https://www.molinstincts.com/sdf-mol-file/L-alanine-sdf-CT1000647025.html
    return _get_sdf_file_path("mol_in.sdf")


@pytest.fixture
def sdf_out_file_path():
    return _get_sdf_file_path("mol_out.sdf")


def test_mol_from_sdf(sdf_in_file_path):
    mol_from_sdf = MolFromSDFTransformer()
    mols = mol_from_sdf.transform(sdf_in_file_path)

    assert_equal(len(mols), 1)
    assert all(isinstance(x, Mol) for x in mols)


def test_mol_to_sdf(mols_list, sdf_out_file_path):
    mol_to_sdf = MolToSDFTransformer(sdf_out_file_path)
    mol_to_sdf.transform(mols_list)

    assert os.path.exists(sdf_out_file_path)


def test_mol_to_and_from_sdf(mols_list, sdf_out_file_path):
    mol_from_sdf = MolFromSDFTransformer()
    mol_to_sdf = MolToSDFTransformer(sdf_out_file_path)

    mol_to_sdf.transform(mols_list)
    mols_list_2 = mol_from_sdf.transform(sdf_out_file_path)

    assert_equal(len(mols_list_2), len(mols_list))
    assert all(isinstance(x, Mol) for x in mols_list_2)


def test_mol_from_sdf_parallel_from_file(sdf_in_file_path):
    mol_from_sdf = MolFromSDFTransformer(n_jobs=2)
    mols = mol_from_sdf.transform(sdf_in_file_path)

    assert_equal(len(mols), 1)
    assert all(isinstance(x, Mol) for x in mols)


def test_mol_from_sdf_parallel_warns_for_raw_text(sdf_in_file_path):
    with open(sdf_in_file_path) as file:
        sdf_text = file.read()

    mol_from_sdf = MolFromSDFTransformer(n_jobs=2)
    with pytest.warns(
        UserWarning,
        match="Parallel SDF reading requires a file path",
    ):
        mols = mol_from_sdf.transform(sdf_text)

    assert_equal(len(mols), 1)
    assert all(isinstance(x, Mol) for x in mols)


def test_mol_from_sdf_parallel_preserves_order(mols_list, tmp_path):
    mols = []
    # add names for verification
    for idx, mol in enumerate(mols_list[:5]):
        mol_copy = Mol(mol)
        name = f"mol_{idx}"
        mol_copy.SetProp("_Name", name)
        mols.append(mol_copy)

    sdf_file_path = tmp_path / "ordered_mols.sdf"
    MolToSDFTransformer(str(sdf_file_path)).transform(mols)

    # test
    sequential_mols = MolFromSDFTransformer().transform(str(sdf_file_path))
    parallel_mols = MolFromSDFTransformer(n_jobs=2).transform(str(sdf_file_path))

    sequential_names = [mol.GetProp("_Name") for mol in sequential_mols]
    parallel_names = [mol.GetProp("_Name") for mol in parallel_mols]

    assert parallel_names == sequential_names


def test_mol_from_sdf_parallel_falls_back_for_older_rdkit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(sdf_module, "_get_rdkit_version", lambda: (2025, 3, 0))
    monkeypatch.setattr(
        sdf_module, "SDMolSupplier", lambda *_args, **_kwargs: [sentinel]
    )

    mol_from_sdf = MolFromSDFTransformer(n_jobs=2)
    with pytest.warns(UserWarning, match="requires RDKit >= 2025.09.1"):
        mols = mol_from_sdf._read_sdf_file("ignored.sdf")

    assert mols == [sentinel]


def test_error_nonexistent_sdf_file():
    mol_from_sdf = MolFromSDFTransformer()
    with pytest.raises(FileNotFoundError):
        mol_from_sdf.transform("nonexistent.sdf")


def _get_sdf_file_path(filename: str) -> str:
    if "tests" in os.listdir():
        return os.path.join("tests", "preprocessing", "input_output", "data", filename)
    if "preprocessing" in os.listdir():
        return os.path.join("preprocessing", "input_output", "data", filename)
    if "input_output" in os.listdir():
        return os.path.join("input_output", "data", filename)
    raise FileNotFoundError(f"File {filename} not found")

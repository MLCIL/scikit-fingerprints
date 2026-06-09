from pathlib import Path

import pytest
from rdkit.Chem import Mol

from skfp.preprocessing import MolFromSDFTransformer


@pytest.fixture(scope="session")
def mordred_test_mols() -> dict[str, Mol]:
    sdf_file = Path(__file__).parent / "references" / "structures.sdf"
    mols = MolFromSDFTransformer().transform(str(sdf_file))  # no hydrogens by default
    mols = {mol.GetProp("_Name"): mol for mol in mols}
    return mols


@pytest.fixture(scope="session")
def mordred_test_mols_hydrogens_3d() -> dict[str, Mol]:
    sdf_file = Path(__file__).parent / "references" / "structures.sdf"
    mols = MolFromSDFTransformer(remove_hydrogens=False).transform(str(sdf_file))
    for mol in mols:
        mol.SetIntProp("conf_id", 0)

    mols = {mol.GetProp("_Name"): mol for mol in mols}
    return mols

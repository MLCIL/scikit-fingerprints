from rdkit.Chem import Mol
from skfp.preprocessing import MolFromInchiTransformer

mol_from_inchi = MolFromInchiTransformer(suppress_warnings=True)


def feasibility_filter_batch(inchis: list[str]) -> list[bool]:
    mols = mol_from_inchi.transform(inchis)
    filter_indicators = [
        filter_mol(inchi, mol) if mol else False
        for inchi, mol in zip(inchis, mols, strict=False)
    ]
    return filter_indicators


def filter_mol(inchi: str, mol: Mol) -> bool:
    from rdkit.Chem import GetMolFrags
    from rdkit.Chem.Crippen import MolLogP
    from rdkit.Chem.Descriptors import MolWt
    from rdkit.Chem.rdMolDescriptors import (
        CalcNumHBA,
        CalcNumHBD,
        CalcNumRotatableBonds,
        CalcTPSA,
    )

    return all(
        [
            len(GetMolFrags(mol)) <= 3,
            len(inchi) < 2000,
            MolWt(mol) <= 2500,
            mol.GetNumAtoms() <= 150,
            CalcNumHBA(mol) <= 20,
            CalcNumHBD(mol) <= 15,
            -10 <= MolLogP(mol) <= 25,
            CalcTPSA(mol) <= 500,
            CalcNumRotatableBonds(mol) <= 60,
        ]
    )

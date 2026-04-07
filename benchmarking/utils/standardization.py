from rdkit.Chem import (
    MolFromInchi,
    MolFromSmiles,
    MolToInchi,
    SanitizeMol,
)
from rdkit.Chem.MolStandardize.rdMolStandardize import CleanupInPlace
from rdkit.Chem.rdmolfiles import MolToSmiles
from rdkit.rdBase import BlockLogs


def inchi_to_inchi_standardize(inchis_list: list[str]) -> list[str | None]:
    inchis_standardized = []
    with BlockLogs():
        for inchi in inchis_list:
            try:
                mol = MolFromInchi(inchi)

                # Kekulize, check valencies, set aromaticity, conjugation and hybridization
                SanitizeMol(mol)

                # remove Hs, disconnect metals, normalize functional groups, reionize
                CleanupInPlace(mol)

                # ensure idempotence by writing back and again
                standardized_inchi = MolToInchi(MolFromInchi(MolToInchi(mol)))

                inchis_standardized.append(standardized_inchi)
            except Exception:
                inchis_standardized.append(None)

    return inchis_standardized


def smiles_to_inchi_convert(smiles_list: list[str]) -> list[str | None]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    inchis_list = []
    for smiles in smiles_list:
        try:
            inchi = MolToInchi(MolFromSmiles(smiles))
            inchis_list.append(inchi)
        except Exception:
            inchis_list.append(None)

    return inchis_list


def inchi_to_smiles_convert(inchis_list: list[str]) -> list[str | None]:
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    smiles_list = []
    for inchi in inchis_list:
        try:
            smiles = MolToSmiles(MolFromInchi(inchi))
            smiles = MolToSmiles(MolFromSmiles(smiles))
            smiles_list.append(smiles)
        except Exception:
            smiles_list.append(None)

    return smiles_list

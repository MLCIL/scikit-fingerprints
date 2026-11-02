"""
Buduje data/prepared/ogbg-molmuv.joblib dokladnie tak, jak zrobilby to download.py.

Odtwarza ogb_solver + build_dataset z src/common/data_v2.py, ale czyta rozpakowane
archiwum OGB wprost, zeby nie ciagnac torcha i torch_geometric tylko po SMILES-y
i indeksy splitow. Kolumna "graph" jest pomijana - pomiar dotyka wylacznie
kolumny "smiles" (timer obejmuje model.embed(batch_df)), a Dataset.labels i tak
ja odrzuca.
"""

import sys

import joblib
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, ".")
from src.common.types import Dataset

RDLogger.DisableLog("rdApp.*")

ROOT = "data/raw/muv"
NAME = "ogbg-molmuv"

df = pd.read_csv(f"{ROOT}/mapping/mol.csv.gz").drop(columns=["mol_id"])
splits = {
    key: pd.read_csv(f"{ROOT}/split/scaffold/{key}.csv.gz", header=None)[0].tolist()
    for key in ("train", "valid", "test")
}

# build_dataset() kanonizuje SMILES-y - bez tego lista illegal_smiles nie dopasuje sie
df["smiles"] = df["smiles"].map(Chem.CanonSmiles)

# download.py woła load(..., load_graphs=True), wiec przygotowany zbior ma kolumne
# "graph"; Dataset.labels bezwarunkowo ja odrzuca, wiec musi istniec. Zadna sciezka
# pomiaru jej nie czyta - placeholder wystarcza i oszczedza torch_geometric.
df["graph"] = None

dataset = Dataset(name=NAME, task="classification", data=df, splits=splits)
joblib.dump(dataset, f"data/prepared/{NAME}.joblib")
print(f"{NAME}: {len(df)} czasteczek, splity {[(k, len(v)) for k, v in splits.items()]}")

# maly zbior do testu instalacji, zeby nie walidowac rurociagu na 30-minutowym przebiegu
small = df.iloc[:500].copy()
small_splits = {"train": list(range(400)), "valid": [], "test": list(range(400, 500))}
joblib.dump(
    Dataset(name=f"{NAME}-smoke", task="classification", data=small, splits=small_splits),
    f"data/prepared/{NAME}-smoke.joblib",
)
print(f"{NAME}-smoke: {len(small)} czasteczek")

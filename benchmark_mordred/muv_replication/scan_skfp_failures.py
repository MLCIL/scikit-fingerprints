"""
Czy MUV (po kanonizacji i po filtrze illegal_smiles) zawiera czasteczki, na
ktorych nasza implementacja rzuca wyjatkiem? Od tego zalezy, czy wrapper musi
odtwarzac zachowanie mordreda (wiersz NaN na czasteczke) czy nie.

Nie jest pomiarem, wiec idzie rownolegle na wszystkich rdzeniach.
"""

import sys

import joblib
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger

sys.path.insert(0, ".")

RDLogger.DisableLog("rdApp.*")

dataset = joblib.load("data/prepared/ogbg-molmuv.joblib")
illegal = [line.strip() for line in open("config/illegal_smiles.txt")]
dataset.filter_out_problematic_molecules(illegal)
smiles = dataset.data["smiles"].str.replace("*", "C", regex=False).tolist()
print(f"po filtrze: {len(smiles)} czasteczek", flush=True)


def check(lo, hi):
    from skfp.fingerprints import NewMordredFingerprint

    fp = NewMordredFingerprint(n_jobs=1, verbose=False)
    mols = [Chem.MolFromSmiles(s) for s in smiles[lo:hi]]
    pairs = [(i, m) for i, m in enumerate(mols, start=lo) if m is not None]
    unparsed = sum(1 for m in mols if m is None)
    try:
        fp.transform([m for _, m in pairs])
        return [], unparsed
    except Exception:
        bad = []
        for i, mol in pairs:
            try:
                fp.transform([mol])
            except Exception as e:
                bad.append((i, smiles[i], mol.GetNumAtoms(), type(e).__name__, str(e)[:100]))
        return bad, unparsed


CHUNK = 400
ranges = [(i, min(i + CHUNK, len(smiles))) for i in range(0, len(smiles), CHUNK)]
out = Parallel(n_jobs=60)(delayed(check)(lo, hi) for lo, hi in ranges)

bad = [b for chunk, _ in out for b in chunk]
unparsed = sum(u for _, u in out)
print(f"nieparsowalnych przez RDKit: {unparsed}")
print(f"wysypuje nasza implementacje: {len(bad)}")
for i, smi, n, exc, msg in bad[:15]:
    print(f"  idx={i:6d} atomow={n:4d} {exc}: {msg}")
    print(f"    {smi[:140]}")

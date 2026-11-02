"""
Buduje zbior ring-rich uzyty w Fazie B scenariusza ExpansionRx.

Pierwsze `--n` czasteczek z MolPILE majacych co najmniej `--min-rings` pierscieni
(`CalcNumRings`). Powstal jako test oczekiwania, ze czasteczki bogate w pierscienie
dadza wyraznie wiekszy speedup - zmierzono 8.19x przy 34.7 atomu i 4.9 pierscienia,
czyli tyle, ile kazdy inny zbior tej wielkosci (bace 7.87x, sider 8.97x).

We wrzesniu zbior powstal dorazne i nie byl odtwarzalny; ten skrypt zamyka te luke.

**UWAGA: to odtwarza regule, nie oryginalny plik co do bajtu.** Kolumna `id` w
`dataset_rings.csv` z wrzesnia siega 123 mln, czyli tamten zbior zostal wyciagniety
z **pelnego MolPILE (222 mln)**, a nie z podzbioru miliona. Z podzbioru dostaniesz
zbior rownowazny pod wzgledem konstrukcji (te same kryteria, ta sama liczebnosc),
ale zlozony z innych czasteczek. Do pomiaru to bez znaczenia - mierzymy klase
czasteczek, nie konkretne sztuki - ale wynik nie bedzie identyczny co do cyfry.

    MOLPILE_CSV=.../molpile_1M.csv python benchmark_mordred/make_rings_dataset.py
"""

import argparse
import os

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--molpile-csv", default=os.environ.get("MOLPILE_CSV", "molpile_1M.csv"))
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--min-rings", type=int, default=4)
    ap.add_argument("--out", default=os.environ.get("BENCH_RINGS_CSV", "dataset_rings.csv"))
    args = ap.parse_args()

    # czytamy strumieniowo: pelny korpus to 42 MB, ale skrypt ma dzialac tak samo
    # na wiekszym pliku, a i tak konczymy po zebraniu --n czasteczek
    rows, seen = [], 0
    for chunk in pd.read_csv(args.molpile_csv, usecols=["SMILES"], chunksize=50_000):
        for smiles in chunk["SMILES"].astype(str):
            seen += 1
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            n_rings = rdMolDescriptors.CalcNumRings(mol)
            if n_rings >= args.min_rings:
                rows.append({"id": seen - 1, "SMILES": smiles, "num_rings": n_rings})
                if len(rows) >= args.n:
                    break
        if len(rows) >= args.n:
            break

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"{len(out)} czasteczek z >= {args.min_rings} pierscieniami "
          f"(przejrzano {seen}), srednio {out.num_rings.mean():.2f} pierscienia")
    print(f"zapisane: {args.out}")


if __name__ == "__main__":
    main()

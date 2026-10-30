# ruff: noqa: INP001
import json
from pathlib import Path

import numpy as np
import polars as pl
from huggingface_hub import hf_hub_download
from rdkit import Chem

from skfp.fingerprints.rdkit_2d_desc import _get_rdkit_2d_descriptors
from skfp.utils import no_rdkit_logs

source = hf_hub_download(
    repo_id="scikit-fingerprints/MolPILE",
    repo_type="dataset",
    filename="molpile_subset_100k.parquet",
)

smiles_list = pl.read_parquet(source, columns=["SMILES"])["SMILES"].to_list()
descriptors = _get_rdkit_2d_descriptors()

values = np.full(
    (len(smiles_list), len(descriptors)),
    np.nan,
    dtype=np.float64,
)

with no_rdkit_logs():
    for i, smiles in enumerate(smiles_list):
        if not isinstance(smiles, str):
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        for j, (_, func) in enumerate(descriptors):
            try:
                values[i, j] = float(func(mol))
            except Exception:  # noqa: BLE001, S112
                continue

cdf_data = {}

for j, (name, _) in enumerate(descriptors):
    finite = np.sort(values[np.isfinite(values[:, j]), j])

    if not finite.size:
        raise ValueError(f"No valid values for descriptor: {name}")

    indices = np.linspace(
        0,
        finite.size - 1,
        min(512, finite.size),
        dtype=int,
    )
    thresholds = np.unique(finite[indices])
    probabilities = np.searchsorted(finite, thresholds, side="right") / finite.size

    cdf_data[name] = [thresholds.tolist(), probabilities.tolist()]

output = (
    Path(__file__).resolve().parents[1]
    / "skfp"
    / "fingerprints"
    / "data"
    / "rdkit_2d_cdf_molpile.json"
)

output.write_text(
    json.dumps(cdf_data, indent=2, allow_nan=False) + "\n",
    encoding="utf-8",
)

print(f"Saved CDF data for {len(cdf_data)} descriptors: {output}")

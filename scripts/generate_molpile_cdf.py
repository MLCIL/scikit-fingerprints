# ruff: noqa: INP001
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from rdkit import Chem

from skfp.fingerprints.rdkit_2d_desc import _get_rdkit_2d_descriptors
from skfp.utils import no_rdkit_logs, run_in_parallel

N_BINS = 512
SAMPLE_SIZE = 10000
BATCH_SIZE = 5000
READ_BATCH_SIZE = 500000
N_JOBS = 64
RANDOM_SEED = 42


def _calculate_batch(smiles_batch):
    """Calculate RDKit 2D descriptors for one batch of SMILES."""
    descriptors = _get_rdkit_2d_descriptors()
    values = np.full(
        (len(smiles_batch), len(descriptors)),
        np.nan,
        dtype=np.float64,
    )

    with no_rdkit_logs():
        for mol_idx, smiles in enumerate(smiles_batch):
            if not isinstance(smiles, str):
                continue

            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            for descriptor_idx, (_, func) in enumerate(descriptors):
                try:
                    values[mol_idx, descriptor_idx] = float(func(mol))
                except Exception:  # noqa: BLE001, S112
                    continue

    return values


def _get_bin_thresholds(values):
    """Choose right-inclusive histogram thresholds from sample values."""
    finite = values[np.isfinite(values)]

    if not finite.size:
        raise ValueError("No finite descriptor values in the sample")

    unique_values = np.unique(finite)
    is_integer = np.all(finite == np.floor(finite))
    minimum = unique_values[0]
    maximum = unique_values[-1]

    max_thresholds = N_BINS - 1

    if is_integer and maximum - minimum + 1 <= max_thresholds:
        return minimum + np.arange(
            int(maximum - minimum) + 1,
            dtype=np.float64,
        )

    if unique_values.size <= max_thresholds:
        return unique_values

    quantiles = np.linspace(0, 1, max_thresholds)
    return np.unique(np.quantile(finite, quantiles, method="higher"))


def _get_batch_histograms(smiles_batch, bin_thresholds):
    """Count values in common, right-inclusive bins for each descriptor."""
    values = _calculate_batch(smiles_batch)
    batch_counts = []
    batch_maxima = np.full(values.shape[1], -np.inf)

    for descriptor_idx, thresholds in enumerate(bin_thresholds):
        finite = values[np.isfinite(values[:, descriptor_idx]), descriptor_idx]

        indices = np.searchsorted(thresholds, finite, side="left")
        counts = np.bincount(indices, minlength=len(thresholds) + 1)

        batch_counts.append(counts)

        if finite.size:
            batch_maxima[descriptor_idx] = finite.max()

    return batch_counts, batch_maxima


def main():  # noqa: PLR0915
    """Generate a histogram approximation of the MolPILE empirical CDF."""
    start_time = perf_counter()

    source = hf_hub_download(
        repo_id="scikit-fingerprints/MolPILE",
        repo_type="dataset",
        filename="molpile_smiles.parquet",
    )

    parquet_file = pq.ParquetFile(source)
    num_molecules = parquet_file.metadata.num_rows
    descriptors = _get_rdkit_2d_descriptors()

    rng = np.random.default_rng(RANDOM_SEED)
    sample_indices = np.sort(
        rng.choice(
            num_molecules,
            size=min(SAMPLE_SIZE, num_molecules),
            replace=False,
        )
    )

    sample_smiles = []
    row_offset = 0

    for batch in parquet_file.iter_batches(
        batch_size=READ_BATCH_SIZE,
        columns=["SMILES"],
    ):
        next_offset = row_offset + batch.num_rows

        first = np.searchsorted(sample_indices, row_offset)
        last = np.searchsorted(sample_indices, next_offset)

        smiles_column = batch.column(0)

        sample_smiles.extend(
            smiles_column[int(idx - row_offset)].as_py()
            for idx in sample_indices[first:last]
        )

        row_offset = next_offset

    sample_values = _calculate_batch(sample_smiles)

    bin_thresholds = [
        _get_bin_thresholds(sample_values[:, descriptor_idx])
        for descriptor_idx in range(len(descriptors))
    ]

    del sample_values

    total_counts = [
        np.zeros(len(thresholds) + 1, dtype=np.int64) for thresholds in bin_thresholds
    ]
    total_maxima = np.full(len(descriptors), -np.inf)

    parquet_file = pq.ParquetFile(source)
    processed_molecules = 0

    for batch in parquet_file.iter_batches(
        batch_size=READ_BATCH_SIZE,
        columns=["SMILES"],
    ):
        smiles_batch = batch.column(0).to_pylist()

        histograms = run_in_parallel(
            _get_batch_histograms,
            smiles_batch,
            n_jobs=N_JOBS,
            batch_size=BATCH_SIZE,
            verbose=0,
            bin_thresholds=bin_thresholds,
        )

        for batch_counts, batch_maxima in histograms:
            for descriptor_idx, counts in enumerate(batch_counts):
                total_counts[descriptor_idx] += counts

            total_maxima = np.maximum(total_maxima, batch_maxima)

        processed_molecules += len(smiles_batch)

        print(
            f"Processed {processed_molecules}/{num_molecules} molecules",
            flush=True,
        )

        del histograms, smiles_batch

    if processed_molecules != num_molecules:
        raise ValueError("Not all molecules were processed")

    cdf_data = {}

    for descriptor_idx, (name, _) in enumerate(descriptors):
        thresholds = bin_thresholds[descriptor_idx]
        counts = total_counts[descriptor_idx]
        count_total = counts.sum()

        if count_total == 0:
            raise ValueError(f"No finite values for descriptor: {name}")

        probabilities = np.cumsum(counts[:-1]) / count_total
        nonempty = counts[:-1] > 0

        thresholds = thresholds[nonempty]
        probabilities = probabilities[nonempty]

        if counts[-1] > 0:
            thresholds = np.append(
                thresholds,
                total_maxima[descriptor_idx],
            )
            probabilities = np.append(probabilities, 1.0)

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

    elapsed_time = perf_counter() - start_time
    minutes = int(elapsed_time // 60)
    seconds = elapsed_time % 60

    print(f"Execution time: {minutes} min {seconds:.2f} s")


if __name__ == "__main__":
    main()

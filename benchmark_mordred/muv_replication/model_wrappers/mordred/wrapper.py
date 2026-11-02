import logging as log
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from mordred import Calculator, descriptors
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from src.common.types import SmilesEmbedder
from src.common.utils import get_embed_cpu_njobs

RDLogger.DisableLog("rdApp.*")


def _descriptor_to_float(value) -> float:
    """Mordred returns error objects for failed descriptors; coerce those to NaN."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _mol_from_smiles(smiles: str, with_3d: bool):
    """Parse SMILES; optionally embed a single 3D conformer for 3D descriptors.

    Module-level so it can be pickled and dispatched to joblib workers.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not with_3d:
        return mol

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.maxAttems = 5000 # From R-MAT
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Retry once with a looser embedding
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        log.warning(f"3D embedding failed for SMILES '{smiles}'")
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        # Optimization is best-effort; coordinates from EmbedMolecule still usable
        pass
    return mol


def _embed_batch(smiles_list: list[str], with_3d: bool, n_jobs: int):
    """Parse/embed a batch of SMILES, in parallel if n_jobs > 1.

    Returns a list of mols (or None for failures) in the same order as
    ``smiles_list``.
    """
    worker = partial(_mol_from_smiles, with_3d=with_3d)

    if n_jobs <= 1 or len(smiles_list) <= 1:
        return [worker(smi) for smi in smiles_list]

    nproc = min(n_jobs, len(smiles_list))
    return Parallel(n_jobs=nproc, backend="loky", batch_size="auto")(
        delayed(worker)(smi) for smi in smiles_list
    )


class MordredEmbedder(SmilesEmbedder):
    """Mordred molecular descriptors as dense embeddings.

    Args:
        model_name: Unique embedder name (must match Hydra model config).
        ignore_3d: If True, compute 2D descriptors only (~1613). If False,
            compute all descriptors including 3D (~1826); SMILES are converted
            to a single ETKDG conformer.
        n_jobs: Worker processes for both conformer embedding and Mordred
            descriptor computation (default: ``get_embed_cpu_njobs()``).
    """

    def __init__(
        self,
        model_name: str = "mordred_2d",
        ignore_3d: bool = True,
        n_jobs: int | None = None,
    ):
        self._model_name = model_name
        self._ignore_3d = ignore_3d
        # Mordred requires nproc >= 1 (unlike sklearn's -1 convention)
        self._n_jobs = get_embed_cpu_njobs() if n_jobs is None else max(1, n_jobs)
        self._calc = Calculator(descriptors, ignore_3D=ignore_3d)
        self._output_dim = len(self._calc.descriptors)

    def _result_to_row(self, result) -> np.ndarray:
        values = np.asarray(
            [_descriptor_to_float(v) for v in result], dtype=np.float64
        )
        # Fill per-descriptor failures; keep full-NaN only for total failures
        if np.isnan(values).all():
            return values
        return np.nan_to_num(values, nan=0.0)

    def forward(self, smiles):
        if hasattr(smiles, "tolist"):
            smiles = smiles.tolist()
        else:
            smiles = list(smiles)

        out = np.full((len(smiles), self._output_dim), np.nan, dtype=np.float64)
        with_3d = not self._ignore_3d

        mols = _embed_batch(smiles, with_3d=with_3d, n_jobs=self._n_jobs)

        valid_indices = []
        valid_mols = []
        for i, (smi, mol) in enumerate(zip(smiles, mols)):
            if mol is None:
                log.error(f"Failed to process SMILES '{smi}'")
                continue
            valid_indices.append(i)
            valid_mols.append(mol)

        if not valid_mols:
            return out

        results = self._calc.map(valid_mols, nproc=self._n_jobs, quiet=True)
        for idx, result in zip(valid_indices, results):
            out[idx] = self._result_to_row(result)
        return out

    @property
    def name(self):
        return self._model_name

    @property
    def device_used(self):
        return "cpu"


def get_embedder(name, **_kwargs):
    name_l = name.lower()
    if "mordred" not in name_l:
        raise ValueError(f"Unknown embedder {name}")
    # "3d" present in mordred_2d3d → all descriptors; mordred_2d → 2D only
    ignore_3d = "3d" not in name_l
    return MordredEmbedder(model_name=name, ignore_3d=ignore_3d)

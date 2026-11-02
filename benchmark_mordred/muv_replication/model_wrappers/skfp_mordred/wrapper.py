"""
Nasz przepisany Mordred (skfp) pod protokolem pomiarowym tego repo.

Lustro `model_wrappers/mordred/wrapper.py`: ten sam kontrakt SmilesEmbedder, ta
sama liczba workerow z `get_embed_cpu_njobs()`, to samo parsowanie SMILES-ow
WEWNATRZ mierzonego `forward()` (helpery `_mol_from_smiles` i `_embed_batch` sa
przeniesione bez zmian), ta sama gesta macierz (n, 1613) float64 na wyjsciu i ta
sama konwencja bledow: wiersz NaN oznacza czasteczke nieobliczalna, ktora harness
usuwa przez `remove_failed_embeddings`.

Rozne jest tylko to, co liczy deskryptory.
"""

import logging as log
from functools import partial

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

from src.common.types import SmilesEmbedder
from src.common.utils import get_embed_cpu_njobs

RDLogger.DisableLog("rdApp.*")


def _mol_from_smiles(smiles: str, with_3d: bool):
    """Przeniesione z model_wrappers/mordred/wrapper.py bez zmian."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if not with_3d:
        return mol

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xF00D
    params.maxAttems = 5000  # From R-MAT
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        log.warning(f"3D embedding failed for SMILES '{smiles}'")
        return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        pass
    return mol


def _embed_batch(smiles_list: list[str], with_3d: bool, n_jobs: int):
    """Przeniesione z model_wrappers/mordred/wrapper.py bez zmian."""
    worker = partial(_mol_from_smiles, with_3d=with_3d)

    if n_jobs <= 1 or len(smiles_list) <= 1:
        return [worker(smi) for smi in smiles_list]

    nproc = min(n_jobs, len(smiles_list))
    return Parallel(n_jobs=nproc, backend="loky", batch_size="auto")(
        delayed(worker)(smi) for smi in smiles_list
    )


class SkfpMordredEmbedder(SmilesEmbedder):
    """Deskryptory Mordreda z naszej implementacji w scikit-fingerprints.

    Args:
        model_name: nazwa embeddera, musi zgadzac sie z configiem Hydry.
        ignore_3d: True -> tylko 2D (1613 cech), jak `mordred_2d`.
        n_jobs: liczba workerow do parsowania i do liczenia deskryptorow
            (domyslnie ``get_embed_cpu_njobs()``, czyli 75% rdzeni).
    """

    def __init__(
        self,
        model_name: str = "skfp_mordred_2d",
        ignore_3d: bool = True,
        n_jobs: int | None = None,
    ):
        from skfp.fingerprints import NewMordredFingerprint

        self._model_name = model_name
        self._ignore_3d = ignore_3d
        self._n_jobs = get_embed_cpu_njobs() if n_jobs is None else max(1, n_jobs)
        self._fp = NewMordredFingerprint(
            use_3D=not ignore_3d, n_jobs=self._n_jobs, verbose=False
        )
        # liczba cech wychodzi z jednej malej czasteczki, tak jak mordred bierze
        # ja z len(calc.descriptors) - poza mierzonym forward()
        probe = self._fp.transform([Chem.MolFromSmiles("CCO")])
        self._output_dim = probe.shape[1]

    def _transform(self, mols) -> np.ndarray:
        """Deskryptory dla listy czasteczek; wiersz NaN dla nieobliczalnych.

        Nasza implementacja przerywa cala paczke wyjatkiem na patologicznych
        czasteczkach (organometaliki: nieskonczona walencja metalu, brak
        zbieznosci LAPACK), a mordred wstawia w takim miejscu placeholder bledu
        i liczy dalej. Zeby obie implementacje oddawaly harnessowi to samo,
        awaryjnie schodzimy do liczenia sztuka po sztuce i zwracamy NaN tylko
        dla winowajcow. Sciezka awaryjna wlacza sie wylacznie po wyjatku, wiec
        na zbiorze bez takich czasteczek nie kosztuje nic.
        """
        try:
            return np.asarray(self._fp.transform(mols), dtype=np.float64)
        except Exception:
            log.warning(f"batch of {len(mols)} raised; chunked fallback")
            return self._fallback(mols)

    def _fallback(self, mols, chunk: int = 500) -> np.ndarray:
        """Awaryjnie paczkami, i dopiero winna paczka sztuka po sztuce.

        Pierwsza wersja szla po calej paczce pojedynczo, kazda czasteczka przez
        pule 96 workerow - na 10 000 czasteczek placilo to rozruch puli dziesiec
        tysiecy razy i pomiar sie nie konczyl. MolPILE ma ~0.09% takich czasteczek,
        wiec ta sciezka bywa uruchamiana i musi byc tania. Pojedyncze czasteczki
        licza sie JEDNYM watkiem, bo pula na jedna czasteczke to sam narzut.
        """
        from skfp.fingerprints import NewMordredFingerprint

        rows = np.full((len(mols), self._output_dim), np.nan, dtype=np.float64)
        single = NewMordredFingerprint(
            use_3D=not self._ignore_3d, n_jobs=1, verbose=False
        )
        for start in range(0, len(mols), chunk):
            part = mols[start : start + chunk]
            try:
                rows[start : start + len(part)] = np.asarray(
                    self._fp.transform(part), dtype=np.float64
                )
                continue
            except Exception:
                pass
            for i, mol in enumerate(part):
                try:
                    rows[start + i] = np.asarray(
                        single.transform([mol]), dtype=np.float64
                    )[0]
                except Exception as e:
                    log.error(f"descriptor computation failed: {type(e).__name__}: {e}")
        return rows

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

        values = self._transform(valid_mols)
        for idx, row in zip(valid_indices, values):
            # ta sama konwencja co w mordred/wrapper.py: pojedyncze NaN-y sa
            # zerowane, caly wiersz NaN zostaje i harness usuwa taka czasteczke
            out[idx] = row if np.isnan(row).all() else np.nan_to_num(row, nan=0.0)
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
    ignore_3d = "3d" not in name_l.replace("2d", "")
    return SkfpMordredEmbedder(model_name=name, ignore_3d=ignore_3d)

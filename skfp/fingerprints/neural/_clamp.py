import functools
from collections.abc import Sequence

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from rdkit.Chem import Mol
from torch import nn

from skfp.bases import BaseFingerprintTransformer
from skfp.fingerprints import ECFPFingerprint, RDKitFingerprint
from skfp.utils import ensure_mols

_CLAMP_HF_REPO = "scikit-fingerprints/clamp"
_CLAMP_HF_FILENAME = "compound_encoder.pt"


class CLAMPCompoundEncoder(nn.Module):
    """
    Two-layer MLP compound encoder from CLAMP [1]_.

    Architecture::

        Input (8192) -> Linear(8192, 4096) -> LayerNorm -> ReLU -> Dropout(0.1)
                     -> Linear(4096, 2048) -> LayerNorm -> ReLU -> Dropout(0.2)
                     -> Linear(2048, 768)
        Output (768)

    Dropout layers are retained to match the original model weights. They have
    no effect at inference time in default ``eval()`` mode

    References
    ----------
    .. [1] `Seidl et al.
        "Enhancing Activity Prediction Models in Drug Discovery with the
        Ability to Understand Human Language"
        International Conference on Machine Learning. PMLR, 2023
        <https://proceedings.mlr.press/v202/seidl23a.html>`_
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8192, 4096),
            nn.LayerNorm(4096, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048, elementwise_affine=False),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 768),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Module-level so lru_cache is keyed only by path, not by instance (avoids memory leaks).
@functools.lru_cache(maxsize=1)
def _load_clamp_model(checkpoint_path: str) -> CLAMPCompoundEncoder:
    """Load pretrained CLAMP compound encoder from a checkpoint file."""
    model = CLAMPCompoundEncoder()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


class CLAMPFingerprint(BaseFingerprintTransformer):
    """
    CLAMP fingerprint.

    CLAMP (Contrastive Language And Molecule Pre-training) uses a pretrained two-layer MLP compound encoder from CLAMP [1]_ to
    transform concatenated ECFP count (4096 bits) and RDKit count (4096 bits)
    fingerprints into 768-dimensional learned embeddings.

    Requires neural optional dependency, installed as scikit-fingerprints[neural]

    Parameters
    ----------
    weights_path : str or None, default=None
        Path to a local pretrained checkpoint file (``.pt``). If ``None``,
        weights are downloaded automatically from the
        ``scikit-fingerprints/clamp`` HuggingFace Hub repository and cached
        in the standard HuggingFace cache directory
        (``~/.cache/huggingface/hub/`` by default).

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 768
        Number of output features, i.e. the CLAMP embedding dimension.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require
        conformers.

    References
    ----------
    .. [1] `Seidl et al.
        "Enhancing Activity Prediction Models in Drug Discovery with the
        Ability to Understand Human Language"
        International Conference on Machine Learning. PMLR, 2023.
        <https://proceedings.mlr.press/v202/seidl23a.html>`_

    Examples
    --------
    >>> from skfp.fingerprints.neural import CLAMPFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = CLAMPFingerprint()
    >>> fp.transform(smiles)  # doctest: +SKIP
    array([...], shape=(4, 768), dtype=float32)
    """

    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "weights_path": [str, None],
    }

    def __init__(
        self,
        weights_path: str | None = None,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=768,
            count=False,
            sparse=False,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )
        self.weights_path = weights_path

    def transform(self, X: Sequence[str | Mol], copy: bool = False) -> np.ndarray:
        """
        Compute CLAMP fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : ndarray of shape (n_samples, 768)
            Array with CLAMP embeddings as float32.
        """
        return super().transform(X, copy=copy)

    def get_model(self) -> CLAMPCompoundEncoder:
        """
        Return the pretrained CLAMP compound encoder.

        Returns
        -------
        encoder : CLAMPCompoundEncoder
            Pretrained ``nn.Module`` in eval mode.
        """
        path = self.weights_path or hf_hub_download(
            repo_id=_CLAMP_HF_REPO, filename=_CLAMP_HF_FILENAME
        )
        return _load_clamp_model(path)

    def get_input_features(self, X: Sequence[str | Mol]) -> np.ndarray:
        """
        Compute CLAMP encoder input features.

        Returns the intermediate 8192-dimensional representation used as input
        to the pretrained encoder in :meth:`transform`: a log-scaled sum of
        ECFP count and RDKit count fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        Returns
        -------
        X : ndarray of shape (n_samples, 8192)
            Array with encoder input features as float32.
        """
        X = ensure_mols(X)
        ecfp = ECFPFingerprint(
            fp_size=8192,
            radius=2,
            use_pharmacophoric_invariants=True,
            include_chirality=True,
            count=True,
        )
        rdkit_fp = RDKitFingerprint(
            fp_size=8192,
            max_path=6,
            num_bits_per_feature=1,
            count=True,
        )
        ecfpc = np.asarray(ecfp.transform(X))
        rdkc = np.asarray(rdkit_fp.transform(X))
        return np.log(1.0 + ecfpc + rdkc).astype(np.float32)

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
        # "Mc+RDKc" preprocessing — see:
        # https://github.com/ml-jku/mhn-react/blob/main/mhnreact/molutils.py#L161-L190
        features = self.get_input_features(X)

        model = self.get_model()
        with torch.inference_mode():
            # np.asarray() ensures ndarray even under a global pandas
            # transform_output config (torch.from_numpy() requires ndarray).
            embeddings = model(torch.from_numpy(np.asarray(features))).numpy()

        return embeddings

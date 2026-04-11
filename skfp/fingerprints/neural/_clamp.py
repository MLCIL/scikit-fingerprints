from collections.abc import Sequence

import numpy as np
import torch
from rdkit.Chem import Mol

from skfp.bases import BaseFingerprintTransformer
from skfp.fingerprints import ECFPFingerprint, RDKitFingerprint
from skfp.fingerprints.neural.utils import _get_weights_path
from skfp.utils import ensure_mols

from ._clamp_model import get_clamp_model

_CLAMP_HF_REPO = "scikit-fingerprints/clamp"
_CLAMP_HF_FILENAME = "compound_encoder.pt"


class CLAMPFingerprint(BaseFingerprintTransformer):
    """
    CLAMP (Contrastive Language And Molecule Pre-training) fingerprint.

    Uses a pretrained two-layer MLP compound encoder from CLAMP [1]_ to
    transform concatenated ECFP count (4096 bits) and RDKit count (4096 bits)
    fingerprints into 768-dimensional learned embeddings.

    Requires PyTorch (``torch``) as an additional dependency.

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

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray:
        X = ensure_mols(X)

        # compute the "Mc+RDKc" input fingerprint as defined in the CLAMP
        # paper: element-wise sum of folded count fingerprints
        # from FCFP-style Morgan (useFeatures=True, useChirality=True) and
        # RDKit FP (maxPath=6, 1 bit per feature), then log(1+x) scaled
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
        ecfpc = ecfp.transform(X)
        rdkc = rdkit_fp.transform(X)
        features = np.log(1.0 + ecfpc + rdkc).astype(np.float32)

        # load model and run inference
        path = self.weights_path or _get_weights_path(
            _CLAMP_HF_REPO, _CLAMP_HF_FILENAME
        )
        model = get_clamp_model(path)

        with torch.no_grad():
            features_tensor = torch.from_numpy(features)
            embeddings = model(features_tensor).numpy()

        return embeddings

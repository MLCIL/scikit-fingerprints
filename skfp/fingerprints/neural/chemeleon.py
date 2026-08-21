from collections.abc import Sequence

import numpy as np
import torch
from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
from chemprop.models import MPNN
from chemprop.nn import BondMessagePassing, MeanAggregation, RegressionFFN
from rdkit.Chem import Mol

from skfp.fingerprints.neural.base_neural_fp_transformer import (
    BaseNeuralFingerprintTransformer,
    DeviceLike,
    TensorLike,
)
from skfp.utils import ensure_mols


class ChemeleonFingerprint(BaseNeuralFingerprintTransformer):
    """
    CheMeleon fingerprint.

    CheMeleon [1]_ uses pretrained message passing neural networks (MPNNs) to generate
    2048-dimensional learned embeddings of molecular graphs.

    Requires neural optional dependency, installed as scikit-fingerprints[neural]

    Parameters
    ----------
    weights_path : str or None, default=None
        Path to a local pretrained checkpoint file (``.pt``). If ``None``,
        weights are downloaded automatically from the
        ``scikit-fingerprints/chemeleon`` HuggingFace Hub repository and cached
        in the standard HuggingFace cache directory
        (``~/.cache/huggingface/hub/`` by default).

    n_jobs : int, default=None
        The number of jobs to run in parallel. :meth:`transform` is parallelized
        over the input molecules when the device is CPU. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        If the device is not CPU, the model runs all processing sequentially,
        regardless of this value.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each forward pass through the model.
        ``None`` means all inputs are processed in a single pass when running
        sequentially. When running with multiple CPU jobs, ``None`` instead
        divides the input into as many equal-sized parts as ``n_jobs``.

    device : str or torch.device, default="cpu"
        Device to use for the model inference. If the device is not CPU,
        the model runs all processing sequentially, regardless of ``n_jobs``.
        Use ``batch_size`` to bound memory usage on such devices.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    Attributes
    ----------
    n_features_out : int = 2048
        Number of output features, i.e. the CheMeleon embedding dimension.

    requires_conformers : bool = False
        This fingerprint uses only 2D molecular graphs and does not require
        conformers.

    References
    ----------
    .. [1] `Burns et al.
        "CheMeleon: Descriptor-based Foundation Model for Molecular Property
        Prediction"
        arXiv:2506.15792, 2025.
        <https://arxiv.org/abs/2506.15792>`_

    Examples
    --------
    >>> from skfp.fingerprints.neural import ChemeleonFingerprint
    >>> smiles = ["O", "CC", "[C-]#N", "CC=O"]
    >>> fp = ChemeleonFingerprint()
    >>> fp.transform(smiles)  # doctest: +SKIP
    array([...], shape=(4, 2048), dtype=float32)
    """

    _HF_REPO = "scikit-fingerprints/chemeleon"
    _HF_FILENAME = "weights.pt"

    def __init__(
        self,
        weights_path: str | None = None,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        device: DeviceLike = "cpu",
        verbose: int | dict = 0,
    ):
        super().__init__(
            n_features_out=2048,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            device=device,
            weights_path=weights_path,
        )

    def transform(self, X: Sequence[str | Mol], copy: bool = False) -> np.ndarray:
        """
        Compute CheMeleon fingerprints.

        Parameters
        ----------
        X : {sequence of str or Mol}
            Sequence containing SMILES strings or RDKit ``Mol`` objects.

        copy : bool, default=False
            Whether to copy input data.

        Returns
        -------
        X : ndarray of shape (n_samples, 2048)
            Array with CheMeleon embeddings as float32.
        """
        return super().transform(X, copy=copy)

    def get_model(self) -> MPNN:
        """
        Return the pretrained CheMeleon model.

        Returns
        -------
        model : MPNN
            Pretrained Chemprop ``MPNN`` in eval mode.
        """
        return super().get_model()

    @classmethod
    def _load_model(cls, path: str) -> MPNN:
        """Load pretrained CheMeleon MPNN from a checkpoint file."""
        chemeleon_mp = torch.load(path, map_location="cpu", weights_only=True)
        mp = BondMessagePassing(**chemeleon_mp["hyper_parameters"])
        mp.load_state_dict(chemeleon_mp["state_dict"])
        model = MPNN(
            message_passing=mp,
            agg=MeanAggregation(),
            # not actually used, fingerprints are taken before the predictor head
            predictor=RegressionFFN(input_dim=mp.output_dim),
        )
        model.eval()
        return model

    def _prepare_input(self, X: Sequence[str | Mol]) -> BatchMolGraph:
        X = ensure_mols(X)
        featurizer = SimpleMoleculeMolGraphFeaturizer()
        return BatchMolGraph([featurizer(mol) for mol in X])

    def _forward_nn(self, X: BatchMolGraph) -> TensorLike:
        model = self.get_model()
        return model.fingerprint(X)

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import scipy.sparse
import torch
from huggingface_hub import hf_hub_download
from joblib import effective_n_jobs
from rdkit.Chem import Mol
from scipy.sparse import csr_array
from tqdm import tqdm

from skfp.bases.base_fp_transformer import BaseFingerprintTransformer
from skfp.utils import run_in_parallel

_MODEL_CACHE: dict[tuple[type, str, str], object] = {}


class BaseNeuralFingerprintTransformer(BaseFingerprintTransformer):
    """
    Base class for neural fingerprints.

    Utilizes PyTorch-based pre-trained neural networks to compute fingerprints.

    This class is not meant to be used directly. If you want to use custom neural fingerprints,
    inherit from this class and implement the ``._prepare_input()`` and ``._forward_nn()`` methods.

    Parameters
    ----------
    n_features_out : int
        Number of output features.

    requires_conformers : bool, default=False
        Whether the fingerprint requires 3D conformations as inputs.

    sparse : bool, default=False
        Whether to return dense NumPy array, or sparse SciPy CSR array.

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
        sequentially. When running with multiple CPU jobs, ``None``
        instead divides the input into as many equal-sized parts as ``n_jobs``.

    verbose : int or dict, default=0
        Controls the verbosity when computing fingerprints.
        If a dictionary is passed, it is treated as kwargs for ``tqdm()``,
        and can be used to control the progress bar.

    device : str or torch.device, default="cpu"
        Device to use for the model inference. If the device is not CPU,
        the model runs all processing sequentially, regardless of ``n_jobs``.
        Use ``batch_size`` to bound memory usage on such devices.

    weights_path : str or None, default=None
        Path to a local pretrained checkpoint file (``.pt``).
    """

    _HF_REPO: str
    _HF_FILENAME: str

    # parameters common for all neural fingerprints
    _parameter_constraints: dict = {
        **BaseFingerprintTransformer._parameter_constraints,
        "device": [str, torch.device],
        "weights_path": [str, None],
    }

    def __init__(
        self,
        n_features_out: int,
        requires_conformers: bool = False,
        sparse: bool = False,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        device: str | torch.device = "cpu",
        weights_path: str | None = None,
    ):
        super().__init__(
            n_features_out=n_features_out,
            requires_conformers=requires_conformers,
            sparse=sparse,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
        )

        self.device = device
        self.weights_path = weights_path

    def get_model(self):
        """
        Get the pre-trained model.

        The return type depends on the subclass — it does not have to be a
        ``torch.nn.Module``, only something ``_forward_nn()`` knows how to call.

        Returns
        -------
        model
            The pretrained model, as constructed by ``_load_model()``, moved
            to ``self.device``, and cached.
        """
        path = self.weights_path or hf_hub_download(
            repo_id=self._HF_REPO, filename=self._HF_FILENAME
        )
        return self._get_cached_model(path, self.device)

    @classmethod
    def _get_cached_model(cls, path: str, device: str | torch.device):
        """
        Get the cached model.

        Models are cached by class, path and device.

        :param path: Path to the model checkpoint file.
        :param device: Device to load the model on.
        :return: The cached model.
        """
        device_str = str(device)
        key = (cls, path, device_str)
        if key in _MODEL_CACHE:
            return _MODEL_CACHE[key]
        model = cls._load_model(path).to(device).eval()
        _MODEL_CACHE[key] = model
        return model

    @classmethod
    @abstractmethod
    def _load_model(cls, path: str):
        """
        Construct the model and load its weights from a local checkpoint file.

        The base class moves the returned model to the requested device, sets
        it to evaluation mode, and caches it — this hook only needs to build
        the model and load the checkpoint.
        """
        raise NotImplementedError

    @abstractmethod
    def _prepare_input(self, X: Sequence[str | Mol]):
        """
        Convert a batch of molecules into model input features.

        The result is passed through :meth:`_to_device` before being handed
        to :meth:`_forward_nn`.
        """
        raise NotImplementedError

    @abstractmethod
    def _forward_nn(self, X) -> torch.Tensor | np.ndarray:
        """
        Run the model forward pass on a device-placed input batch.

        Called with the output of :meth:`_prepare_input` after it has been
        moved to ``self.device`` by :meth:`_to_device`.
        """
        raise NotImplementedError

    def _to_device(self, X):
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X).to(self.device)
        if hasattr(X, "to"):
            return X.to(self.device)
        if isinstance(X, dict) and all(hasattr(x, "to") for x in X.values()):
            return {k: v.to(self.device) for k, v in X.items()}
        raise RuntimeError("Cannot convert NN model to requested device")

    def _calculate_fingerprint(self, X: Sequence[str | Mol]) -> np.ndarray | csr_array:
        with torch.no_grad():
            X_input = self._prepare_input(X)
            X_input = self._to_device(X_input)
            X_output = self._forward_nn(X_input)
        if isinstance(X_output, torch.Tensor):
            X_output = X_output.detach().cpu().numpy()
        return X_output

    def transform(
        self, X: Sequence[str | Mol], copy: bool = False
    ) -> np.ndarray | csr_array:
        """
        Compute fingerprints. Output shape depends on the inheriting class.

        Parameters
        ----------
        X : {sequence, array-like} of shape (n_samples,)
            Sequence containing SMILES strings or RDKit ``Mol`` objects. Depending on
            the implementation in the inheriting class, it may require using ``Mol``
            objects with computed conformations and with ``conf_id`` property set.

        copy : bool, default=False
            Copy the input X or not.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, any)
            Array with fingerprints.
        """
        self._validate_params()

        if copy:
            X = deepcopy(X)

        n_jobs = effective_n_jobs(self.n_jobs)

        if torch.device(self.device).type == "cpu" and n_jobs > 1:
            results = run_in_parallel(
                self._calculate_fingerprint,
                data=X,
                n_jobs=n_jobs,
                batch_size=self.batch_size,
                verbose=self.verbose,
            )
        elif self.verbose:
            results = [self._calculate_fingerprint([mol]) for mol in tqdm(X)]
        elif self.batch_size is not None:
            results = [
                self._calculate_fingerprint(X[i : i + self.batch_size])
                for i in range(0, len(X), self.batch_size)
            ]
        else:
            results = self._calculate_fingerprint(X)

        if isinstance(results, (np.ndarray, csr_array)):
            return results
        else:
            return scipy.sparse.vstack(results) if self.sparse else np.vstack(results)

import numpy as np
import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from skfp.fingerprints.neural import base_neural_fp_transformer as base_module
from skfp.fingerprints.neural.base_neural_fp_transformer import (
    BaseNeuralFingerprintTransformer,
)
from skfp.utils import ensure_mols

"""
BaseNeuralFingerprintTransformer is an abstract base class (ABC), so we test it
through a minimal concrete subclass.
"""

N_FEATURES_IN = 4
N_FEATURES_OUT = 3

# recorded by RecordingNeuralFingerprint, only meaningful for n_jobs=1, since
# with multiprocessing the subprocesses append to their own copies
BATCH_SIZES: list[int] = []


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(N_FEATURES_IN, N_FEATURES_OUT)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass through the dummy encoder."""
        return self.linear(x)


class DummyNeuralFingerprint(BaseNeuralFingerprintTransformer):
    _HF_REPO = "scikit-fingerprints/dummy"
    _HF_FILENAME = "dummy.pt"

    def __init__(
        self,
        weights_path: str | None = None,
        n_jobs: int | None = None,
        batch_size: int | None = None,
        verbose: int | dict = 0,
        device: str | torch.device = "cpu",
    ):
        super().__init__(
            n_features_out=N_FEATURES_OUT,
            n_jobs=n_jobs,
            batch_size=batch_size,
            verbose=verbose,
            device=device,
            weights_path=weights_path,
        )

    @classmethod
    def _load_model(cls, path: str) -> DummyEncoder:
        model = DummyEncoder()
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        return model

    def _prepare_input(self, X) -> np.ndarray:
        mols = ensure_mols(X)
        features = [
            [
                mol.GetNumAtoms(),
                mol.GetNumBonds(),
                mol.GetNumHeavyAtoms(),
                len(mol.GetRingInfo().AtomRings()),
            ]
            for mol in mols
        ]
        return np.array(features, dtype=np.float32)

    def _forward_nn(self, X: torch.Tensor) -> torch.Tensor:
        return self.get_model()(X)


class RecordingNeuralFingerprint(DummyNeuralFingerprint):
    def _prepare_input(self, X) -> np.ndarray:
        BATCH_SIZES.append(len(X))
        return super()._prepare_input(X)


@pytest.fixture(scope="session")
def weights_path(tmp_path_factory) -> str:
    torch.manual_seed(0)
    path = tmp_path_factory.mktemp("weights") / "dummy.pt"
    torch.save(DummyEncoder().state_dict(), path)
    return str(path)


@pytest.fixture(autouse=True)
def _clear_recordings():
    BATCH_SIZES.clear()


@pytest.fixture
def clear_model_cache():
    base_module._MODEL_CACHE.clear()
    yield
    base_module._MODEL_CACHE.clear()


def get_second_device() -> str | None:
    """Return an available non-CPU device, or None if there is none."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


@pytest.mark.parametrize("n_jobs", [1, 2])
def test_base_neural_verbose(n_jobs, weights_path, smiles_list, capsys):
    fp = DummyNeuralFingerprint(weights_path=weights_path, n_jobs=n_jobs, verbose=True)
    fp.transform(smiles_list)

    output = capsys.readouterr().err
    assert "100%" in output
    assert "it/s" in output


def test_base_neural_batch_size_splits_input(weights_path, smiles_list):
    batch_size = 8
    fp = RecordingNeuralFingerprint(
        weights_path=weights_path, batch_size=batch_size, n_jobs=1
    )
    fp.transform(smiles_list)

    n_expected = -(-len(smiles_list) // batch_size)  # ceil division
    assert len(BATCH_SIZES) == n_expected
    assert sum(BATCH_SIZES) == len(smiles_list)
    assert max(BATCH_SIZES) <= batch_size


def test_base_neural_no_batch_size_uses_single_pass(weights_path, smiles_list):
    fp = RecordingNeuralFingerprint(
        weights_path=weights_path, batch_size=None, n_jobs=1
    )
    fp.transform(smiles_list)

    assert len(BATCH_SIZES) == 1
    assert BATCH_SIZES[0] == len(smiles_list)


@pytest.mark.usefixtures("clear_model_cache")
def test_base_neural_model_is_cached(weights_path):
    fp_1 = DummyNeuralFingerprint(weights_path=weights_path)
    fp_2 = DummyNeuralFingerprint(weights_path=weights_path)

    # the cache is global, so separate instances must share a single model
    assert fp_1.get_model() is fp_1.get_model()
    assert fp_1.get_model() is fp_2.get_model()
    assert len(base_module._MODEL_CACHE) == 1


@pytest.mark.usefixtures("clear_model_cache")
def test_base_neural_model_cache_separates_devices(weights_path):
    second_device = get_second_device()
    if second_device is None:
        pytest.skip("no non-CPU device available")

    model_cpu = DummyNeuralFingerprint(
        weights_path=weights_path, device="cpu"
    ).get_model()
    model_other = DummyNeuralFingerprint(
        weights_path=weights_path, device=second_device
    ).get_model()

    assert model_cpu is not model_other
    assert next(model_cpu.parameters()).device.type == "cpu"
    assert next(model_other.parameters()).device.type == second_device
    # the cached CPU model must not be moved by the second instance
    assert next(model_cpu.parameters()).device.type == "cpu"


def test_base_neural_uses_weights_path_without_download(
    weights_path, smiles_list, monkeypatch
):
    def fail_download(*args, **kwargs):
        raise AssertionError("hf_hub_download() must not be called for weights_path")

    monkeypatch.setattr(base_module, "hf_hub_download", fail_download)

    fp = DummyNeuralFingerprint(weights_path=weights_path)
    assert fp.transform(smiles_list).shape == (len(smiles_list), N_FEATURES_OUT)


@pytest.mark.usefixtures("clear_model_cache")
def test_base_neural_downloads_weights_when_path_not_given(weights_path, monkeypatch):
    calls = []

    def fake_download(repo_id, filename):
        calls.append((repo_id, filename))
        return weights_path

    monkeypatch.setattr(base_module, "hf_hub_download", fake_download)

    fp = DummyNeuralFingerprint(weights_path=None)
    fp.get_model()

    assert calls == [
        (DummyNeuralFingerprint._HF_REPO, DummyNeuralFingerprint._HF_FILENAME)
    ]


def test_base_neural_to_device_ndarray(weights_path):
    fp = DummyNeuralFingerprint(weights_path=weights_path)
    X = fp._to_device(np.zeros((2, N_FEATURES_IN), dtype=np.float32))

    assert isinstance(X, torch.Tensor)
    assert X.device.type == "cpu"


def test_base_neural_to_device_tensor(weights_path):
    fp = DummyNeuralFingerprint(weights_path=weights_path)
    X = fp._to_device(torch.zeros(2, N_FEATURES_IN))

    assert isinstance(X, torch.Tensor)
    assert X.device.type == "cpu"


def test_base_neural_to_device_dict_of_tensors(weights_path):
    fp = DummyNeuralFingerprint(weights_path=weights_path)
    X = fp._to_device({"a": torch.zeros(2), "b": torch.ones(2)})

    assert set(X) == {"a", "b"}
    assert all(value.device.type == "cpu" for value in X.values())

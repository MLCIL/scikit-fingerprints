from collections.abc import Callable

import numpy as np
from sklearn.datasets import make_blobs


def get_metric_data_type(metric: str | Callable) -> str:
    """
    Pick the kind of features a given metric is defined for, based on its name.
    """
    name = metric if isinstance(metric, str) else getattr(metric, "__name__", "")
    if "binary" in name:
        return "binary"
    elif "count" in name:
        return "counts"
    else:
        return "continuous"


def get_data_inside_ad(
    n_train: int = 1000, n_test: int = 100, data_type: str = "continuous"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Training data as a wide blob, test data as a tight blob at its center, i.e.
    firmly inside the applicability domain.
    """
    n_features = 2 if data_type == "continuous" else 256

    X_train = _make_blobs(
        n_train, cluster_std=10.0, center_box=10.0, n_features=n_features
    )
    X_test = _make_blobs(n_test, cluster_std=0.1, center_box=0.1, n_features=n_features)

    return _convert(X_train, X_test, data_type)


def get_data_outside_ad(
    n_train: int = 1000, n_test: int = 100, data_type: str = "continuous"
) -> tuple[np.ndarray, np.ndarray]:
    """
    Training data as a wide blob, test data far away from it, i.e. firmly outside
    the applicability domain.
    """
    n_features = 2 if data_type == "continuous" else 256

    X_train = _make_blobs(
        n_train, cluster_std=10.0, center_box=10.0, n_features=n_features
    )

    if data_type == "binary":
        X_train, _ = _to_binary(X_train, X_train)
        # complement of the training rows gives maximally dissimilar bit vectors
        X_test = 1 - X_train[:n_test]
        _check_data_type(X_train, data_type)
        _check_data_type(X_test, data_type)
        return X_train, X_test

    X_test = X_train[:n_test] + 100.0
    return _convert(X_train, X_test, data_type)


def _make_blobs(n_samples: int, cluster_std: float, center_box: float, n_features: int):
    X, _ = make_blobs(
        n_samples=n_samples,
        centers=1,
        cluster_std=cluster_std,
        center_box=(-center_box, center_box),
        random_state=0,
        n_features=n_features,
    )
    return X


def _convert(X_train: np.ndarray, X_test: np.ndarray, data_type: str) -> tuple:
    if data_type == "binary":
        X_train, X_test = _to_binary(X_train, X_test)
    elif data_type == "counts":
        X_train, X_test = _to_counts(X_train, X_test)
    elif data_type == "continuous":
        pass
    else:
        raise ValueError(f"Unrecognized data type {data_type}")

    _check_data_type(X_train, data_type)
    _check_data_type(X_test, data_type)
    return X_train, X_test


def _to_binary(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # threshold comes from the training data, so that both subsets are comparable
    threshold = np.percentile(X_train, 25)
    return (X_train > threshold).astype(int), (X_test > threshold).astype(int)


def _to_counts(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # shift data to have non-negative values
    shift = X_train.min()

    X_train = np.rint(np.clip(X_train - shift, 0, None)).astype(int)
    X_test = np.rint(np.clip(X_test - shift, 0, None)).astype(int)
    return X_train, X_test


def _check_data_type(X: np.ndarray, data_type: str) -> None:
    if data_type == "binary":
        assert np.isin(X, (0, 1)).all(), "binary data must contain only 0s and 1s"
    elif data_type == "counts":
        assert np.issubdtype(X.dtype, np.integer), "counts must be integers"
        assert X.min() >= 0, f"counts must be non-negative, got minimum {X.min()}"

import itertools
from collections.abc import Callable, Sequence

import numpy as np
from joblib import effective_n_jobs
from sklearn.utils.parallel import Parallel, delayed
from tqdm.auto import tqdm


class ProgressParallel(Parallel):
    """
    A more verbose version of ``joblib.Parallel``, which outputs progress bar.

    Parameters
    ----------
    tqdm_settings: dict or None, default=None
        Settings to use for the ``tqdm()`` progress bar.
    """

    def __init__(self, *args, tqdm_settings: dict | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if tqdm_settings is None:
            tqdm_settings = {}
        self._tqdm_settings: dict = tqdm_settings

    def __call__(self, *args, **kwargs):  # noqa: D102
        with tqdm(**self._tqdm_settings) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self) -> None:  # noqa: D102
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def run_in_parallel(
    func: Callable,
    data: Sequence,
    n_jobs: int | None = None,
    batch_size: int | None = None,
    single_element_func: bool = False,
    flatten_results: bool = False,
    verbose: int | dict = 0,
    **kwargs,
) -> list:
    """
    Run a function in parallel on provided data in batches, using joblib.

    Results are returned in the same order as input data. ``func`` function must take
    batch of data, e.g. list of integers, not a single integer.

    If ``func`` returns lists, the result will be a list of lists. To get a flat list
    of results, use ``flatten_results=True``.

    Note that progress bar for ``verbose`` option tracks processing of data batches,
    not individual data points.

    Parameters
    ----------
    func : Callable
        The function to run in parallel. It must take only a single argument,
        a batch of data.

    data : {sequence, array-like} of shape (n_samples,)
        Sequence containing data to process.

    n_jobs : int, default=None
        The number of jobs to run in parallel. ``None`` means 1 unless in a
        :obj:`joblib.parallel_backend` context. ``-1`` means using all processors.
        See scikit-learn documentation on ``n_jobs`` for more details.

    batch_size : int, default=None
        Number of inputs processed in each batch. ``None`` divides input data into
        equal-sized parts, as many as ``n_jobs``.

    single_element_func : bool, default=False
        If True, single element will be passed to ``func``, rather than batches
        of data. This way, non-batched functions can be parallelized. When this
        option is set, ``batch_size`` will be ignored.

    flatten_results : bool, default=False
        Whether to flatten the results, e.g. to change list of lists of integers
        into a list of integers.

    verbose : int or dict, default=0
        Controls the verbosity. If higher than zero, progress bar will be shown,
        tracking the processing of batches. If ``dict`` object is provided,
        it will be used to configure the ``tqdm`` progress bar.

    **kwargs : dict
        parameters specific to the function passed in the ``func`` parameter

    Returns
    -------
    X : list of length (n_samples,)
        The processed data. If the processing function returns lists, this will be
        a list of lists.

    Examples
    --------
    >>> from skfp.utils import run_in_parallel
    >>> func = lambda X: [x + 1 for x in X]
    >>> data = list(range(10))
    >>> run_in_parallel(func, data, n_jobs=-1, batch_size=1)
    [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
    >>> run_in_parallel(func, data, n_jobs=-1, batch_size=1, flatten_results=True)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    >>> func_single = lambda x: x + 1
    >>> run_in_parallel(func_single, data, n_jobs=-1, single_element_func=True)
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    """
    n_jobs = effective_n_jobs(n_jobs)

    if batch_size is None:
        batch_size = max(len(data) // n_jobs, 1)
    elif batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if single_element_func:
        data_batch_gen = (d for d in data)
        num_batches = len(data)
    else:
        data_batch_gen = (
            data[i : i + batch_size] for i in range(0, len(data), batch_size)
        )
        num_batches = np.ceil(len(data) / batch_size)

    if isinstance(verbose, int):
        tqdm_settings = {
            "total": num_batches,
            "disable": verbose == 0,
        }
    elif isinstance(verbose, dict):
        tqdm_settings = verbose.copy()
        tqdm_settings["total"] = num_batches
        tqdm_settings["disable"] = verbose.get("disable", False)
    else:
        raise TypeError(
            f"The verbose argument must be int or dict, got {type(verbose)}"
        )

    if tqdm_settings["disable"]:
        parallel = Parallel(n_jobs=n_jobs)
    else:
        parallel = ProgressParallel(n_jobs=n_jobs, tqdm_settings=tqdm_settings)

    results = parallel(
        delayed(func)(data_batch, **kwargs) for data_batch in data_batch_gen
    )

    if flatten_results:
        results = list(itertools.chain.from_iterable(results))

    return results

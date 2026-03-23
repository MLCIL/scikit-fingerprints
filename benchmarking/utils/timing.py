import timeit
from collections.abc import Callable
from typing import Any

import numpy as np


def measure_time(
    func: Callable[[Any], Any], data: Any, label: str = None, iterations: int = 5
) -> tuple[float, float]:
    """
    Measure the average execution time of a function over N_REPEATS.
    """
    if label:
        print(f"Benchmarking {label}...")
    timer = timeit.Timer(lambda: func(data))
    times = timer.repeat(repeat=iterations, number=1)
    return np.mean(times), np.std(times)

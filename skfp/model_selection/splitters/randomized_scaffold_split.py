from collections.abc import Sequence
from numbers import Integral
from typing import Any

import numpy as np
from numpy.random import RandomState
from rdkit.Chem import Mol
from sklearn.utils._param_validation import Interval, RealNotInt, validate_params
from sklearn.utils.validation import check_random_state

from skfp.model_selection.splitters.scaffold_split import _create_scaffold_sets
from skfp.model_selection.splitters.utils import (
    ensure_nonempty_subset,
    split_additional_data,
    validate_train_test_split_sizes,
    validate_train_valid_test_split_sizes,
)
from skfp.utils.functions import get_data_from_indices


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "use_csk": ["boolean"],
        "return_indices": ["boolean"],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def randomized_scaffold_train_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    test_size: float | None = None,
    use_csk: bool = False,
    return_indices: bool = False,
    random_state: int | RandomState | None = None,
):
    """
    Split using randomized groups of Bemis-Murcko scaffolds.

    This split uses randomly partitioned groups of Bemis-Murcko molecular scaffolds [1]_
    for splitting. This is a nondeterministic variant of scaffold split, introduced in
    the MoleculeNet [2]_ paper. It aims to verify the model generalization to new scaffolds,
    as an approximation to the time split, while also allowing multiple train-test splits.

    By default, core structure scaffolds are used (following RDKit), which include atom
    types. Original Bemis-Murcko approach uses the cyclic skeleton (CSK) of a molecule,
    replacing all atoms by carbons. It is also known as CSK [3]_, and can be used with
    `use_csk` parameter.

    This approach is known to have certain limitations. In particular, molecules with
    no rings will not get a scaffold, resulting in them being grouped together regardless
    of their structure.

    This variant is nondeterministic, and the scaffolds are randomly shuffled before
    being assigned to subsets (test set is created fist). This approach is also known
    as "balanced scaffold split", and typically leads to more optimistic evaluation than
    regular, deterministic scaffold split [4]_.

    If ``train_size`` and ``test_size`` are integers, they must sum up to the ``data``
    length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data, e.g. labels.

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set
        to 1 - test_size. If test_size is also None, it will be set to 0.8.

    test_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set
        to 1 - train_size. If train_size is also None, it will be set to 0.2.

    use_csk: bool, default=False
        Whether to use the molecule cyclic skeleton (CSK), instead of the core
        structure scaffold.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state : int, RandomState instance or None, default=0
        Seed for random number generator or random state that would be used for
        shuffling the scaffolds.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-test subsets of provided arrays. First two are lists of SMILES
        strings or RDKit ``Mol`` objects, depending on the input type. If `return_indices`
        is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        <https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks>`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        <https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning>`_

    .. [3] `Bemis-Murcko scaffolds and their variants
        <https://github.com/rdkit/rdkit/discussions/6844>`_

    .. [4] `R. Sun, H. Dai, A. Wei Yu
        "Does GNN Pretraining Help Molecular Representation?"
        Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html>`_

    Examples
    --------
    >>> from skfp.model_selection.splitters import randomized_scaffold_train_test_split
    >>> smiles = ['c1ccccc1', 'C1CCCCC1', 'CCO', 'CCN', 'CCCl', 'CCBr', 'CCI', 'CCF']
    >>> train_smiles, test_smiles = randomized_scaffold_train_test_split(
    ...     smiles, train_size=6, test_size=2, random_state=42
    ... )
    >>> train_smiles
    ['CCO', 'CCN', 'CCCl', 'CCBr', 'CCI', 'CCF']
    """
    train_size, test_size = validate_train_test_split_sizes(
        train_size, test_size, len(data)
    )

    scaffold_sets = _create_scaffold_sets(data, use_csk)
    rng = np.random.default_rng(check_random_state(random_state))
    rng.shuffle(scaffold_sets)

    train_idxs: list[int] = []
    test_idxs: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < test_size:
            test_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, test_idxs
        )
        return train_subset, test_subset, *additional_data_split
    else:
        return train_subset, test_subset


@validate_params(
    {
        "data": ["array-like"],
        "additional_data": ["tuple"],
        "train_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "valid_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "test_size": [
            Interval(RealNotInt, 0, 1, closed="neither"),
            Interval(Integral, 1, None, closed="left"),
            None,
        ],
        "use_csk": ["boolean"],
        "return_indices": ["boolean"],
        "random_state": ["random_state"],
    },
    prefer_skip_nested_validation=True,
)
def randomized_scaffold_train_valid_test_split(
    data: Sequence[str | Mol],
    *additional_data: Sequence,
    train_size: float | None = None,
    valid_size: float | None = None,
    test_size: float | None = None,
    use_csk: bool = False,
    return_indices: bool = False,
    random_state: int | RandomState | None = None,
):
    """
    Split using randomized groups of Bemis-Murcko scaffolds.

    This split uses randomly partitioned groups of Bemis-Murcko molecular scaffolds [1]_
    for splitting. This is a nondeterministic variant of scaffold split, introduced in
    the MoleculeNet [2]_ paper. It aims to verify the model generalization to new scaffolds,
    as an approximation to the time split, while also allowing multiple train-test splits.

    By default, core structure scaffolds are used (following RDKit), which include atom
    types. Original Bemis-Murcko approach uses the cyclic skeleton (CSK) of a molecule,
    replacing all atoms by carbons. It is also known as CSK [3]_, and can be used with
    `use_csk` parameter.

    This approach is known to have certain limitations. In particular, molecules with
    no rings will not get a scaffold, resulting in them being grouped together regardless
    of their structure.

    This variant is nondeterministic, and the scaffolds are randomly shuffled before
    being assigned to subsets (in order: test, valid, train). This approach is also known
    as "balanced scaffold split", and typically leads to more optimistic evaluation than
    regular, deterministic scaffold split [4]_.

    If ``train_size``, ``valid_size`` and ``test_size`` are integers, they must sum up
    to the ``data`` length. If they are floating numbers, they must sum up to 1.

    Parameters
    ----------
    data : sequence
        A sequence representing either SMILES strings or RDKit ``Mol`` objects.

    additional_data: sequence
        Additional sequences to be split alongside the main data, e.g. labels.

    train_size : float, default=None
        The fraction of data to be used for the train subset. If None, it is set
        to 1 - test_size - valid_size. If valid_size is not provided, train_size
        is set to 1 - test_size. If train_size, test_size and valid_size aren't
        set, train_size is set to 0.8.

    valid_size : float, default=None
        The fraction of data to be used for the test subset. If None, it is set
        to 1 - train_size - valid_size. If train_size, test_size and valid_size
        aren't set, train_size is set to 0.1.

    test_size : float, default=None
        The fraction of data to be used for the validation subset. If None, it is
        set to 1 - train_size - valid_size. If valid_size is not provided, test_size
        is set to 1 - train_size. If train_size, test_size and valid_size aren't set,
        test_size is set to 0.1.

    use_csk: bool, default=False
        Whether to use the molecule cyclic skeleton (CSK), instead of the core
        structure scaffold.

    return_indices : bool, default=False
        Whether the method should return the input object subsets, i.e. SMILES strings
        or RDKit ``Mol`` objects, or only the indices of the subsets instead of the data.

    random_state : int, RandomState instance or None, default=0
        Seed for random number generator or random state that would be used for
        shuffling the scaffolds.

    Returns
    -------
    subsets : tuple[list, list, ...]
        Tuple with train-valid-test subsets of provided arrays. First three are lists of
        SMILES strings or RDKit ``Mol`` objects, depending on the input type. If
        `return_indices` is True, lists of indices are returned instead of actual data.

    References
    ----------
    .. [1] `Bemis, G. W., & Murcko, M. A.
        "The properties of known drugs. 1. Molecular frameworks."
        Journal of Medicinal Chemistry, 39(15), 2887-2893.
        <https://www.researchgate.net/publication/14493474_The_Properties_of_Known_Drugs_1_Molecular_Frameworks>`_

    .. [2] `Z. Wu, B. Ramsundar, E. N. Feinberg, J. Gomes, C. Geniesse, A. S. Pappu, K. Leswing, V. Pande
        "MoleculeNet: A Benchmark for Molecular Machine Learning."
        Chemical Science, 9(2), 513-530.
        <https://www.researchgate.net/publication/314182452_MoleculeNet_A_Benchmark_for_Molecular_Machine_Learning>`_

    .. [3] `Bemis-Murcko scaffolds and their variants
        <https://github.com/rdkit/rdkit/discussions/6844>`_

    .. [4] `R. Sun, H. Dai, A. Wei Yu
        "Does GNN Pretraining Help Molecular Representation?"
        Advances in Neural Information Processing Systems 35 (NeurIPS 2022).
        <https://proceedings.neurips.cc/paper_files/paper/2022/hash/4ec360efb3f52643ac43fda570ec0118-Abstract-Conference.html>`_

    Examples
    --------
    >>> from skfp.model_selection.splitters import randomized_scaffold_train_valid_test_split
    >>> smiles = ['c1ccccc1', 'C1CCCCC1', 'CCO', 'CCN', 'CCCl', 'CCBr', 'CCI', 'CCF']
    >>> train_smiles, valid_smiles, test_smiles = randomized_scaffold_train_valid_test_split(
    ...     smiles, train_size=6, valid_size=1, test_size=1, random_state=42
    ... )
    >>> train_smiles
    ['CCO', 'CCN', 'CCCl', 'CCBr', 'CCI', 'CCF']
    """
    train_size, valid_size, test_size = validate_train_valid_test_split_sizes(
        train_size, valid_size, test_size, len(data)
    )

    scaffold_sets = _create_scaffold_sets(data, use_csk)
    rng = np.random.default_rng(check_random_state(random_state))
    rng.shuffle(scaffold_sets)

    train_idxs: list[int] = []
    valid_idxs: list[int] = []
    test_idxs: list[int] = []

    for scaffold_set in scaffold_sets:
        if len(test_idxs) < test_size:
            test_idxs.extend(scaffold_set)
        elif len(valid_idxs) < valid_size:
            valid_idxs.extend(scaffold_set)
        else:
            train_idxs.extend(scaffold_set)

    ensure_nonempty_subset(train_idxs, "train")
    ensure_nonempty_subset(valid_idxs, "validation")
    ensure_nonempty_subset(test_idxs, "test")

    if return_indices:
        train_subset = train_idxs
        valid_subset = valid_idxs
        test_subset = test_idxs
    else:
        train_subset = get_data_from_indices(data, train_idxs)
        valid_subset = get_data_from_indices(data, valid_idxs)
        test_subset = get_data_from_indices(data, test_idxs)

    if additional_data:
        additional_data_split: list[Sequence[Any]] = split_additional_data(
            list(additional_data), train_idxs, valid_idxs, test_idxs
        )
        return train_subset, valid_subset, test_subset, *additional_data_split
    else:
        return train_subset, valid_subset, test_subset

import os
from collections.abc import Iterator

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from skfp.datasets.utils import fetch_splits

from .adme import (
    load_b3db_classification,
    load_b3db_regression,
    load_bioavailability_ma,
    load_caco2_wang,
    load_clearance_microsome_az,
    load_cyp1a2_veith,
    load_cyp2c9_substrate_carbonmangels,
    load_cyp2c9_veith,
    load_cyp2c19_veith,
    load_cyp2d6_substrate_carbonmangels,
    load_cyp2d6_veith,
    load_cyp3a4_substrate_carbonmangels,
    load_cyp3a4_veith,
    load_half_life_obach,
    load_hia_hou,
    load_hlm,
    load_pampa_approved_drugs,
    load_pampa_ncats,
    load_pgp_broccatelli,
    load_ppbr_az,
    load_rlm,
    load_solubility_aqsoldb,
    load_vdss_lombardo,
)
from .hts import load_sarscov2_3clpro_diamond, load_sarscov2_vitro_touret
from .tox import (
    load_ames,
    load_carcinogens_lagunin,
    load_dili,
    load_herg,
    load_herg_central_at_1um,
    load_herg_central_at_10um,
    load_herg_central_inhib,
    load_herg_karim,
    load_ld50_zhu,
    load_skin_reaction,
)

TDC_DATASET_NAMES = [
    # ADME
    "b3db_classification",
    "b3db_regression",
    "bioavailability_ma",
    "caco2_wang",
    "clearance_microsome_az",
    "cyp1a2_veith",
    "cyp2c19_veith",
    "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith",
    "cyp2d6_substrate_carbonmangels",
    "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels",
    "cyp3a4_veith",
    "half_life_obach",
    "hia_hou",
    "hlm",
    "pampa_ncats",
    "pampa_approved_drugs",
    "pgp_broccatelli",
    "ppbr_az",
    "rlm",
    "solubility_aqsoldb",
    "vdss_lombardo",
    # HTS
    "sarscov2_3clpro_diamond",
    "sarscov2_vitro_touret",
    # Toxicity
    "ames",
    "carcinogens_lagunin",
    "dili",
    "herg",
    "herg_central_at_10um",
    "herg_central_at_1um",
    "herg_central_inhib",
    "herg_karim",
    "ld50_zhu",
    "skin_reaction",
]

TDC_DATASET_NAME_TO_LOADER_FUNC = {
    # ADME
    "b3db_classification": load_b3db_classification,
    "b3db_regression": load_b3db_regression,
    "bioavailability_ma": load_bioavailability_ma,
    "caco2_wang": load_caco2_wang,
    "clearance_microsome_az": load_clearance_microsome_az,
    "cyp1a2_veith": load_cyp1a2_veith,
    "cyp2c19_veith": load_cyp2c19_veith,
    "cyp2c9_substrate_carbonmangels": load_cyp2c9_substrate_carbonmangels,
    "cyp2c9_veith": load_cyp2c9_veith,
    "cyp2d6_substrate_carbonmangels": load_cyp2d6_substrate_carbonmangels,
    "cyp2d6_veith": load_cyp2d6_veith,
    "cyp3a4_substrate_carbonmangels": load_cyp3a4_substrate_carbonmangels,
    "cyp3a4_veith": load_cyp3a4_veith,
    "half_life_obach": load_half_life_obach,
    "hia_hou": load_hia_hou,
    "hlm": load_hlm,
    "pampa_approved_drugs": load_pampa_approved_drugs,
    "pampa_ncats": load_pampa_ncats,
    "pgp_broccatelli": load_pgp_broccatelli,
    "ppbr_az": load_ppbr_az,
    "rlm": load_rlm,
    "solubility_aqsoldb": load_solubility_aqsoldb,
    "vdss_lombardo": load_vdss_lombardo,
    # HTS
    "sarscov2_3clpro_diamond": load_sarscov2_3clpro_diamond,
    "sarscov2_vitro_touret": load_sarscov2_vitro_touret,
    # Toxicity
    "ames": load_ames,
    "carcinogens_lagunin": load_carcinogens_lagunin,
    "dili": load_dili,
    "herg": load_herg,
    "herg_central_at_10um": load_herg_central_at_10um,
    "herg_central_at_1um": load_herg_central_at_1um,
    "herg_central_inhib": load_herg_central_inhib,
    "herg_karim": load_herg_karim,
    "ld50_zhu": load_ld50_zhu,
    "skin_reaction": load_skin_reaction,
}


@validate_params(
    {
        "subset": [
            None,
            StrOptions({"ADME", "HTS", "Toxicity"}),
            list,
        ],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tdc_benchmark(
    subset: str | list[str] | None = None,
    data_dir: str | os.PathLike | None = None,
    as_frames: bool = False,
    verbose: bool = False,
) -> Iterator[tuple[str, pd.DataFrame]] | Iterator[tuple[str, list[str], np.ndarray]]:
    """
    Load the TDC benchmark datasets.

    TDC [1]_ datasets are varied molecular property prediction tasks. Scaffold split is
    recommended for all of them. The tasks are split into 3 different groups:

    - ADME - absorbtion, distribution, metabolism, excertion
    - HTS - high-throughput screening
    - Toxicity - toxicity

    For more details, see loading functions for particular datasets. Allowed individual
    dataset names are listed below. Dataset names are also returned (case-sensitive).

    ADME group:

    - "b3db_classification"
    - "b3db_regression"
    - "bioavailability_ma"
    - "caco2_wang"
    - "clearance_microsome_az"
    - "cyp1a2_veith"
    - "cyp2c19_veith"
    - "cyp2c9_substrate_carbonmangels"
    - "cyp2c9_veith"
    - "cyp2d6_substrate_carbonmangels"
    - "cyp2d6_veith"
    - "cyp3a4_substrate_carbonmangels"
    - "cyp3a4_veith"
    - "half_life_obach"
    - "hia_hou"
    - "hlm"
    - "pampa_ncats"
    - "pampa_approved_drugs"
    - "pgp_broccatelli"
    - "ppbr_az"
    - "rlm"
    - "solubility_aqsoldb"
    - "vdss_lombardo"

    High throughput screening (HTS) group:

    - "sarscov2_3clpro_diamond"
    - "sarscov2_vitro_touret"

    Toxicity group:

    - "ames"
    - "carcinogens_lagunin"
    - "dili"
    - "herg"
    - "herg_central_at_10um"
    - "herg_central_at_1um"
    - "herg_central_inhib"
    - "herg_karim"
    - "ld50_zhu"
    - "skin_reaction"

    Parameters
    ----------
    subset : {None, "ADME", "HTS", "Toxicity"}, default=None
        If ``None``, returns all datasets. String loads only a given subset of all
        datasets. Alternatively the subset can contain names of individual datasets.
        List of strings loads only datasets with given names.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frames : bool, default=False
        If True, returns the raw DataFrame for each dataset. Otherwise, returns SMILES
        as a list of strings, and labels as a NumPy array for each dataset.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : generator of pd.DataFrame or tuples (list[str], np.ndarray)
        Loads and returns datasets with a generator. Returned types depend on the
        ``as_frame`` parameter, either:
        - Pandas DataFrame with columns: "SMILES", "label"
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    dataset_names = _subset_to_dataset_names(subset)

    dataset_functions = [
        TDC_DATASET_NAME_TO_LOADER_FUNC[name] for name in dataset_names
    ]

    if as_frames:
        # generator of tuples (dataset_name, DataFrame)
        datasets = (
            (dataset_name, load_function(data_dir, as_frame=True, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )
    else:
        # generator of tuples (dataset_name, SMILES, y)
        datasets = (
            (dataset_name, *load_function(data_dir, as_frame=False, verbose=verbose))
            for dataset_name, load_function in zip(
                dataset_names, dataset_functions, strict=False
            )
        )

    return datasets


@validate_params(
    {
        "dataset_name": [StrOptions(set(TDC_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tdc_dataset(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_frame: bool = False,
    verbose: bool = False,
) -> pd.DataFrame | tuple[list[str], np.ndarray]:
    """
    Load TDC dataset by name.

    Loads a given dataset from TDC [1]_ benchmark by its name. This is a proxy for
    easier benchmarking, that avoids looking for individual functions.

    Dataset names here are the same as returned by `load_tdc_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_frame : bool, default=False
        If True, returns the raw DataFrame with columns "SMILES" and labels
        (dataset-dependent). Otherwise, returns SMILES as list of strings, and
        labels as a NumPy array (shape and type are dataset-dependent).

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : pd.DataFrame or tuple(list[str], np.ndarray)
        Depending on the ``as_frame`` argument, one of:
        - Pandas DataFrame with columns depending on the dataset
        - tuple of: list of strings (SMILES), NumPy array (labels)

    References
    ----------
    .. [1] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    loader_func = TDC_DATASET_NAME_TO_LOADER_FUNC[dataset_name]
    return loader_func(data_dir, as_frame, verbose)


@validate_params(
    {
        "dataset_name": [StrOptions(set(TDC_DATASET_NAMES))],
        "data_dir": [None, str, os.PathLike],
        "as_frame": ["boolean"],
        "verbose": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_tdc_splits(
    dataset_name: str,
    data_dir: str | os.PathLike | None = None,
    as_dict: bool = False,
    verbose: bool = False,
) -> tuple[list[int], list[int], list[int]] | dict[str, list[int]]:
    """
    Load pre-generated dataset splits from the TDC benchmark.

    TDC [1]_ uses precomputed scaffold split with 80/10/10% split between train/valid/test
    subsets. Those splits are widely used in literature and allow for
    a realistic estimate of model performance on new data.

    Dataset names here are the same as returned by `load_tdc_benchmark` function,
    and are case-sensitive.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to loads splits for.

    data_dir : {None, str, path-like}, default=None
        Path to the root data directory. If ``None``, currently set scikit-learn directory
        is used, by default `$HOME/scikit_learn_data`.

    as_dict : bool, default=False
        If True, returns the splits as dictionary with keys "train", "valid" and "test",
        and index lists as values. Otherwise, returns three lists with splits indexes.

    verbose : bool, default=False
        If True, progress bar will be shown for downloading or loading files.

    Returns
    -------
    data : tuple(list[int], list[int], list[int]) or dict
        Depending on the `as_dict` argument, one of:
        - three lists of integer indexes
        - dictionary with "train", "valid" and "test" keys, and values as lists with
        splits indexes

    References
    ----------
    .. [1] `Huang, Kexin, et al.
        "Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development"
        Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks, 2021
        <https://openreview.net/forum?id=8nvgnORnoWr>`_
    """
    splits = fetch_splits(
        data_dir,
        dataset_name=f"TDC_{dataset_name}",
        filename=f"tdc_{dataset_name.lower()}_splits.json",
        verbose=verbose,
    )
    if as_dict:
        return splits
    else:
        return splits["train"], splits["valid"], splits["test"]


def _subset_to_dataset_names(subset: str | list[str] | None) -> list[str]:
    # map given subset (e.g. "ADME", "HTS" or "Toxicity") to list of dataset names
    # for appropriate TDC datasets

    adme_names = [
        "b3db_classification",
        "b3db_regression",
        "bioavailability_ma",
        "caco2_wang",
        "clearance_microsome_az",
        "cyp1a2_veith",
        "cyp2c19_veith",
        "cyp2c9_substrate_carbonmangels",
        "cyp2c9_veith",
        "cyp2d6_substrate_carbonmangels",
        "cyp2d6_veith",
        "cyp3a4_substrate_carbonmangels",
        "cyp3a4_veith",
        "half_life_obach",
        "hia_hou",
        "hlm",
        "pampa_ncats",
        "pampa_approved_drugs",
        "pgp_broccatelli",
        "ppbr_az",
        "rlm",
        "solubility_aqsoldb",
        "vdss_lombardo",
    ]

    hts_names = [
        "sarscov2_3clpro_diamond",
        "sarscov2_vitro_touret",
    ]

    tox_names = [
        "ames",
        "carcinogens_lagunin",
        "dili",
        "herg",
        "herg_central_at_10um",
        "herg_central_at_1um",
        "herg_central_inhib",
        "herg_karim",
        "ld50_zhu",
        "skin_reaction",
    ]

    all_dataset_names = adme_names + hts_names + tox_names

    if subset is None:
        dataset_names = all_dataset_names
    elif subset == "ADME":
        dataset_names = adme_names
    elif subset == "HTS":
        dataset_names = hts_names
    elif subset == "Toxicity":
        dataset_names = tox_names
    elif isinstance(subset, (list, set, tuple)):
        for name in subset:
            if name not in all_dataset_names:
                raise ValueError(
                    f"Dataset name '{name}' not recognized among TDC datasets"
                )
        dataset_names = subset
    else:
        raise ValueError(
            f'Value "{subset}" for subset not recognized, must be one of: '
            f'"adme", "hts", or "tox", alternatively '
            f"be a list of strings with dataset names from TDC to load"
        )

    return dataset_names

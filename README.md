<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logos/skfp-logo-horizontal-text-white.png">
    <source media="(prefers-color-scheme: light)" srcset="docs/logos/skfp-logo-horizontal-text-black.png">
    <img alt="scikit-fingerprints" src="docs/logos/skfp-logo-no-text.png">
  </picture>

  <br/>
  <br/>

  <p>
    <a href="https://badge.fury.io/py/scikit-fingerprints"><img src="https://badge.fury.io/py/scikit-fingerprints.svg" alt="PyPI version"></a>
    <a href="https://pepy.tech/project/scikit-fingerprints"><img src="https://static.pepy.tech/badge/scikit-fingerprints" alt="Downloads"></a>
    <a href="https://pypi.org/project/scikit-fingerprints/"><img src="https://img.shields.io/pypi/dm/scikit-fingerprints" alt="Monthly downloads"></a>
    <a href="LICENSE.md"><img src="https://img.shields.io/badge/license-MIT-blue" alt="License"></a>
    <a href="https://pypi.org/project/scikit-fingerprints/"><img src="https://img.shields.io/pypi/pyversions/scikit-fingerprints.svg" alt="Python versions"></a>
    <a href="https://github.com/scikit-fingerprints/scikit-fingerprints/graphs/contributors"><img src="https://img.shields.io/github/contributors/scikit-fingerprints/scikit-fingerprints" alt="Contributors"></a>
  </p>

  <p><strong>The scikit-learn compatible library for molecular fingerprints and chemoinformatics.</strong></p>

  <p>
    Easily and efficiently compute molecular fingerprints, molecular filters, distances &amp; similarity measures, and more.
  </p>

  <p>Go from SMILES to production-grade chemoinformatics ML pipelines in a few lines of code.</p>


  <p>
    <a href="https://scikit-fingerprints.readthedocs.io/latest/"><strong>Documentation</strong></a> &middot;
    <a href="https://scikit-fingerprints.readthedocs.io/latest/examples.html"><strong>Examples & tutorials</strong></a> &middot;
    <a href="https://scikit-fingerprints.readthedocs.io/latest/api_reference.html"><strong>API Reference</strong></a> &middot;
    <a href="https://www.sciencedirect.com/science/article/pii/S2352711024003145"><strong>Publication</strong></a>
</p>
</div>

---

## Table of Contents

  - [Install](#install)
  - [Quickstart](#quickstart)
  - [Key features](#key-features)
  - [Tutorials](#tutorials)
  - [Publications and citing](#publications-and-citing)
  - [Contributing](#contributing)
  - [License](#license)

---

## Install

You can install from PyPI, using `pip` or `uv`.

```bash
pip install scikit-fingerprints
```

If you want to use neural fingerprints (embeddings from pretrained neural networks), install optional dependencies with:

```bash
pip install "scikit-fingerprints[neural]"
```

See [INSTALL.md](INSTALL.md) for more details.

If you need bleeding-edge features and don't mind potentially unstable or undocumented functionalities, you can also install directly from GitHub:

```bash
pip install git+https://github.com/MLCIL/scikit-fingerprints.git
```

Python versions from 3.10 to 3.13 are supported on all major operating systems.
Tests are run on Linux Ubuntu, Windows, and macOS.

## Quickstart

Simply input SMILES strings into the molecular fingerprint instance:

```python
from skfp.fingerprints import ECFPFingerprint

smiles = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

fp = ECFPFingerprint()
X = fp.transform(smiles)  # SMILES in, NumPy array out
```

Build a full molecular ML pipeline with scikit-learn:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union

from skfp.datasets.moleculenet import load_clintox
from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
from skfp.metrics import extract_pos_proba, multioutput_auroc_score
from skfp.model_selection import scaffold_train_test_split
from skfp.preprocessing import MolFromSmilesTransformer

smiles, y = load_clintox()
smiles_train, smiles_test, y_train, y_test = scaffold_train_test_split(
    smiles, y, test_size=0.2
)

pipeline = make_pipeline(
    MolFromSmilesTransformer(),
    make_union(ECFPFingerprint(count=True), MACCSFingerprint()),
    RandomForestClassifier(random_state=0),
)
pipeline.fit(smiles_train, y_train)

y_pred_proba = extract_pos_proba(pipeline.predict_proba(smiles_test))
print(f"AUROC: {multioutput_auroc_score(y_test, y_pred_proba):.2%}")
```

---

## Key features

- **[Molecular fingerprints](https://scikit-fingerprints.readthedocs.io/latest/modules/fingerprints.html)**
  - over 30, e.g. ECFP, Avalon, MACCS, Mordred, PubChem
  - all with a uniform `.transform()` API

- **[Molecular filters](https://scikit-fingerprints.readthedocs.io/latest/modules/filters.html)**
  - over 30, e.g. Lipinski Rule of 5, PAINS, REOS
  - both substructural and physicochemical

- **[Similarity & distance measures](https://scikit-fingerprints.readthedocs.io/latest/modules/distances.html)**
  - 14 measures, e.g. Tanimoto, Dice, MCS
  - compatible with kNN, UMAP, HDBSCAN, and other distance-based models
  - efficient bulk similarity distribution computation

- **[Applicability domain checks](https://scikit-fingerprints.readthedocs.io/latest/modules/applicability_domain.html)**
  - 11 methods, e.g. kNN, centroid distance, TOPKAT
  - evaluate the reliability of algorithms for new molecules

- **[Benchmark datasets](https://scikit-fingerprints.readthedocs.io/latest/modules/datasets.html)**
  - MoleculeNet, Therapeutics Data Commons, MoleculeACE, and LRGB
  - train-test splits built-in

- **Native scikit-learn integration**
  - use `Pipeline`, `FeatureUnion`, `GridSearchCV`, and more
  - build, save, and deploy ML pipelines for chemoinformatics

- **Other features**
  - fast and efficient: parallelized, sparse matrices support, C++ RDKit under the hood
  - efficient hyperparameter tuning with fingerprints caching
  - MIT licensed, permissive academic and commercial use

---

## Tutorials

Step-by-step Jupyter notebooks, both for learning and deploying production-grade features:

1. [Introduction to scikit-fingerprints](examples/01_skfp_introduction.ipynb)
2. [Fingerprint types](examples/02_fingerprint_types.ipynb)
3. [Molecular pipelines](examples/03_pipelines.ipynb)
4. [Conformers and 3D fingerprints](examples/04_conformers.ipynb)
5. [Hyperparameter tuning](examples/05_hyperparameter_tuning.ipynb)
6. [Dataset splits](examples/06_dataset_splits.ipynb)
7. [Datasets and benchmarking](examples/07_datasets_and_benchmarking.ipynb)
8. [Similarity and distance metrics](examples/08_similarity_and_distance_metrics.ipynb)
9. [Molecular filters](examples/09_molecular_filters.ipynb)
10. [Molecular clustering](examples/10_molecular_clustering.ipynb)

---

## Publications and citing

Publications using scikit-fingerprints:
1. [J. Adamczyk, W. Czech "Molecular Topological Profile (MOLTOP) -- Simple and Strong Baseline for Molecular Graph Classification" ECAI 2024](https://ebooks.iospress.nl/doi/10.3233/FAIA240663)
2. [J. Adamczyk, P. Ludynia "Scikit-fingerprints: easy and efficient computation of molecular fingerprints in Python" SoftwareX](https://www.sciencedirect.com/science/article/pii/S2352711024003145)
3. [J. Adamczyk, P. Ludynia, W. Czech "Molecular Fingerprints Are Strong Models for Peptide Function Prediction" ArXiv preprint](https://arxiv.org/abs/2501.17901)
4. [J. Adamczyk "Towards Rational Pesticide Design with Graph Machine Learning Models for Ecotoxicology" CIKM 2025](https://dl.acm.org/doi/abs/10.1145/3746252.3761660)
5. [J. Adamczyk, J. Poziemski, F. Job, M. Król, M. Makowski "MolPILE - large-scale, diverse dataset for molecular representation learning" ArXiv preprint](https://arxiv.org/abs/2509.18353)
6. [J. Adamczyk, J. Poziemski, P. Siedlecki "Evaluating machine learning models for predicting pesticide toxicity to honey bees" Ecotoxicology and Environmental Safety 2026](https://www.sciencedirect.com/science/article/pii/S0147651326001983)
7. [M. Fitzner et al. "BayBE: a Bayesian Back End for experimental planning in the low-to-no-data regime" RSC Digital Discovery](https://pubs.rsc.org/en/content/articlehtml/2025/dd/d5dd00050e)
8. [J. Xiong et al. "Bridging 3D Molecular Structures and Artificial Intelligence by a Conformation Description Language"](https://www.biorxiv.org/content/10.1101/2025.05.07.652440v1.abstract)
9. [S. Mavlonazarova et al. "Untargeted Metabolomics Reveals Organ-Specific and Extraction-Dependent Metabolite Profiles in Endemic Tajik Species Ferula violacea Korovin" bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.08.24.671964v1)

If you use scikit-fingerprints in your work, please cite our publication in
[SoftwareX (open access)](https://www.sciencedirect.com/science/article/pii/S2352711024003145):

```bibtex
@article{scikit_fingerprints,
   title = {Scikit-fingerprints: Easy and efficient computation of molecular fingerprints in Python},
   author = {Jakub Adamczyk and Piotr Ludynia},
   journal = {SoftwareX},
   volume = {28},
   pages = {101944},
   year = {2024},
   issn = {2352-7110},
   doi = {https://doi.org/10.1016/j.softx.2024.101944},
   url = {https://www.sciencedirect.com/science/article/pii/S2352711024003145},
}
```

Also available as a [preprint on ArXiv](https://arxiv.org/abs/2407.13291).

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) and [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details.

## License

MIT -- see [LICENSE.md](LICENSE.md) for details.

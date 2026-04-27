Welcome to scikit-fingerprints documentation!
=============================================

A scikit-learn compatible library for efficient computation of molecular fingerprints.

.. image:: logos/skfp-logo-text-white.png
   :align: center
   :width: 45%



Main features:

- scikit-learn compatible
- feature-rich, with >30 fingerprints
- parallelization
- sparse matrix support
- commercial-friendly MIT license

Easiest way to get started is to install it with pip:

.. code-block:: console

    pip install scikit-fingerprints

Optional dependencies are available as extras. For example, to enable neural fingerprints (CLAMP),
install the ``neural`` extra (this pulls in PyTorch ``torch``):

.. code-block:: console

    pip install "scikit-fingerprints[neural]"

See :doc:`installation` for details.

Then, see the quickstart below, or go for more detailed :doc:`examples`.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   installation
   examples
   api_reference

----------
Quickstart
----------

Most fingerprints are based on molecular graphs (topological, 2D-based), and you can use SMILES
input directly:

.. code-block:: python

    from skfp.fingerprints import AtomPairFingerprint

    smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

    atom_pair_fingerprint = AtomPairFingerprint()

    X = atom_pair_fingerprint.transform(smiles_list)
    print(X)

For fingerprints using conformers (conformational, 3D-based), you need to create molecules first
and compute conformers. Those fingerprints have ``requires_conformers`` attribute set
to ``True``.

.. code-block:: python

    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
    from skfp.fingerprints import WHIMFingerprint

    smiles_list = ["O=S(=O)(O)CCS(=O)(=O)O", "O=C(O)c1ccccc1O"]

    mol_from_smiles = MolFromSmilesTransformer()
    conf_gen = ConformerGenerator()
    fp = WHIMFingerprint()
    print(fp.requires_conformers)  # True

    mols_list = mol_from_smiles.transform(smiles_list)
    mols_list = conf_gen.transform(mols_list)

    X = fp.transform(mols_list)
    print(X)

You can also use scikit-learn functionalities like pipelines, feature unions
etc. to build complex workflows. Popular datasets, e.g. from MoleculeNet benchmark,
can be loaded directly.

.. code-block:: python

    from skfp.datasets.moleculenet import load_clintox
    from skfp.metrics import multioutput_auroc_score, extract_pos_proba
    from skfp.model_selection import scaffold_train_test_split
    from skfp.fingerprints import ECFPFingerprint, MACCSFingerprint
    from skfp.preprocessing import MolFromSmilesTransformer

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline, make_union


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

    y_pred_proba = pipeline.predict_proba(smiles_test)
    y_pred_proba = extract_pos_proba(y_pred_proba)
    auroc = multioutput_auroc_score(y_test, y_pred_proba)
    print(f"AUROC: {auroc:.2%}")


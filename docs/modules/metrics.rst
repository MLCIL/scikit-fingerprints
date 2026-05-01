=======
Metrics
=======

Functions for computing model quality metrics.

.. automodule:: skfp.metrics

=========================================================

.. py:currentmodule:: skfp.metrics

General metrics:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    spearman_correlation

Multioutput metrics:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    multioutput_accuracy_score
    multioutput_auprc_score
    multioutput_auroc_score
    multioutput_balanced_accuracy_score
    multioutput_cohen_kappa_score
    multioutput_f1_score
    multioutput_matthews_corr_coef
    multioutput_mean_absolute_error
    multioutput_mean_squared_error
    multioutput_precision_score
    multioutput_recall_score
    multioutput_root_mean_squared_error
    multioutput_spearman_correlation

Virtual screening metrics:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    bedroc_score
    enrichment_factor
    rie_score

Utility functions:

.. autosummary::
    :nosignatures:
    :toctree: generated/

    extract_pos_proba

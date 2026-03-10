=================================
TDC
=================================

.. automodule:: skfp.datasets.tdc
    :exclude-members: load_tdc_benchmark, load_tdc_splits

=========================================================

.. py:currentmodule:: skfp.datasets.tdc

TDC molecular datasets

.. autosummary::
    :nosignatures:
    :toctree: generated/

    load_tdc_benchmark
    load_tdc_splits

ADME dataset loaders
------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    adme.load_b3db_classification
    adme.load_b3db_regression
    adme.load_bioavailability_ma
    adme.load_caco2_wang
    adme.load_clearance_microsome_az
    adme.load_cyp1a2_veith
    adme.load_cyp2c19_veith
    adme.load_cyp2c9_substrate_carbonmangels
    adme.load_cyp2c9_veith
    adme.load_cyp2d6_substrate_carbonmangels
    adme.load_cyp2d6_veith
    adme.load_cyp3a4_substrate_carbonmangels
    adme.load_cyp3a4_veith
    adme.load_half_life_obach
    adme.load_hia_hou
    adme.load_hlm
    adme.load_pampa_approved_drugs
    adme.load_pampa_ncats
    adme.load_pgp_broccatelli
    adme.load_ppbr_az
    adme.load_rlm
    adme.load_solubility_aqsoldb
    adme.load_vdss_lombardo

HTS dataset loaders
-----------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    hts.load_sarscov2_3clpro_diamond
    hts.load_sarscov2_vitro_touret

Toxicity dataset loaders
----------------------------

.. autosummary::
    :nosignatures:
    :toctree: generated/

    tox.load_ames
    tox.load_dili
    tox.load_carcinogens_lagunin
    tox.load_herg
    tox.load_herg_central_at_1um
    tox.load_herg_central_at_10um
    tox.load_herg_central_inhib
    tox.load_herg_karim
    tox.load_ld50_zhu
    tox.load_skin_reaction
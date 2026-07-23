===========
Descriptors
===========

Modules for computing molecular descriptors.

.. automodule:: skfp.descriptors

=========================================================

.. py:currentmodule:: skfp.descriptors

Charge descriptors (based on atomic electric charges):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    atomic_partial_charges

Constitutional descriptors (based on atomic composition):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    average_molecular_weight
    bond_count
    element_atom_count
    element_counts
    heavy_atom_count
    molecular_weight
    number_of_rings
    number_of_rotatable_bonds
    total_atom_count

Topological descriptors (based on graph topology/structure):

.. autosummary::
    :nosignatures:
    :toctree: generated/

    average_wiener_index
    balaban_j_index
    burden_matrix
    diameter
    graph_distance_index
    hall_kier_alpha
    kappa1_index
    kappa2_index
    kappa3_index
    petitjean_index
    polarity_number
    radius
    wiener_index
    zagreb_index_m1
    zagreb_index_m2

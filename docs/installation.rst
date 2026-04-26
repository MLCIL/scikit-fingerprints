============
Installation
============

This page covers base installation and optional dependencies (“extras”). For the same information
in Markdown (e.g. for GitHub), see ``INSTALL.md`` in the repository root.

Base installation
=================

Install from PyPI:

.. code-block:: console

    pip install scikit-fingerprints


Optional dependencies (extras)
==============================

Some functionality depends on optional dependencies. Install them via extras.

Neural fingerprints
---------------------------

To use neural fingerprints (e.g. CLAMP), install the ``neural`` extra (this installs PyTorch ``torch``):

.. code-block:: console

    pip install "scikit-fingerprints[neural]"

If you need a specific PyTorch build (CPU/CUDA), install PyTorch following the official instructions
and then install scikit-fingerprints (with or without the extra).


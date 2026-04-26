============
Installation
============

This page covers base installation and optional dependencies (“extras”).

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

If you need a specific PyTorch build (CPU/CUDA), install PyTorch following the official instructions. We highly recommend using uv for this to avoid dependency conflicts.


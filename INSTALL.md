# Installation

This page covers base installation and optional dependencies (“extras”).

## Base installation

```bash
pip install scikit-fingerprints
```

## Optional dependencies (extras)

Some functionality depends on optional dependencies. Install them via extras:

### Neural fingerprints (CLAMP): `neural`

```bash
pip install "scikit-fingerprints[neural]"
```

This installs **PyTorch** (`torch`).

If you need a specific PyTorch build (CPU/CUDA), follow the official PyTorch installation instructions
first, then install scikit-fingerprints.

For example, CPU wheels via the official PyTorch index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "scikit-fingerprints[neural]"
```


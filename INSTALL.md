# Installation

This page covers base installation and optional dependencies ("extras").

## Base installation

```bash
pip install scikit-fingerprints
```

## Optional dependencies

Some functionalities depend on optional dependencies. They are installed as package "extras", as shown below.

### Neural fingerprints (CLAMP): `neural`

```bash
pip install "scikit-fingerprints[neural]"
```

This installs **PyTorch** (`torch`).

If you need a specific PyTorch build (CPU/CUDA), install PyTorch following the official instructions. We highly recommend using [uv](https://docs.astral.sh/uv/) for this to avoid dependency conflicts.

For example, CPU wheels via the official PyTorch index:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu "scikit-fingerprints[neural]"
```

### Mordred descriptors: `mordredcommunity`

```bash
pip install "scikit-fingerprints[mordredcommunity]"
```

This installs **mordredcommunity**. `MordredFingerprint` computes Mordred descriptors with an optimized built-in implementation by default, and needs this extra only with `implementation="mordredcommunity"`. Use it to reproduce results of earlier scikit-fingerprints versions, which computed Mordred descriptors with `mordredcommunity`.

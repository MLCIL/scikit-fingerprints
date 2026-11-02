#!/bin/bash

if $INSTALL_DEP; then
  pip install scikit-fingerprints
fi

export HYDRA_EXPERIMENT=skfp_mordred

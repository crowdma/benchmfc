#!/bin/bash

## Train on MFC and Test on MFCUnseen, MFCPacking, MFCEvolving

python src/eval.py \
    experiment=mlp-ember-test \
    task_name=mlp-ember-MFC-0.0 \
    data_name=MFCUnseen \
    pack_ratio=0.0 \
    ckpt_path=logs/mlp-ember-MFC-0.0/<path>/<to>/<epoch_*.ckpt>
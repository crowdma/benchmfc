#!/bin/bash

python detect/mlp_ember.py \
    --model-file logs/mlp-ember-MFC-0.0/train/runs/*/checkpoints/best.pt \
    --data-name MFCUnseen \
    --pack-ratio 1.0
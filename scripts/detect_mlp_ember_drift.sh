#!/bin/bash

# python detect/mlp_ember.py \
#     --model-file logs/mlp-ember-MFC-0.0/train/runs/2023-07-30_11-54-21/checkpoints/best.pt \
#     --data-name MFCUnseen

# python detect/mlp_ember.py \
#     --model-file logs/mlp-ember-MFC-0.0/train/runs/2023-07-30_11-54-21/checkpoints/best.pt \
#     --data-name MFCEvolving

# python detect/mlp_ember.py \
#     --model-file logs/mlp-ember-MFC-0.0/train/runs/2023-07-30_11-54-21/checkpoints/best.pt \
#     --data-name MFCPacking \
#     --pack-ratio 1.0

python detect/mlp_ember.py \
    --model-file logs/mlp-ember-MFC-0.0/train/runs/2023-07-30_11-54-21/checkpoints/best.pt \
    --data-name MFCUnseenPacking \
    --pack-ratio 1.0
#!/bin/bash

# python src/train_gbdt.py \
#     --data-name MFC \
#     --do-wandb

# python src/train_gbdt.py \
#     --data-name MFCUnseen \
#     --do-wandb

# python src/train_gbdt.py \
#     --data-name MFCEvolving \
#     --do-wandb

python src/train_gbdt.py \
    --data-name MFCPacking \
    --pack-ratio 1.0 \
    --do-wandb
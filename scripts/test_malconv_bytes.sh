#! /bin/bash

## MalConv | Train and Test on MFCEvolving, MFCPacking, MFCUnseen
python src/eval.py \
    experiment=malconv-bytes-test \
    task_name=malconv-bytes-MFC-0.0 \
    data_name=MFC \
    pack_ratio=0.0 \
    ckpt_path=logs/malconv-bytes-MFC-0.0/<path>/<to>/<epoch_*.ckpt>
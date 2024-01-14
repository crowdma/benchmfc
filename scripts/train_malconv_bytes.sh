#!/bin/bash

python src/train.py \
    experiment=malconv-bytes-train \
    task_name=malconv-bytes-MFC-0.0 \
    data_name=MFC \
    pack_ratio=0.0 

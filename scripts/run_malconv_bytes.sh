#!/bin/bash
# CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFC
# CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCUnseen
# CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCEvolving
# CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=1.0
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.9
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.8
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.7
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.6
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.5
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.4
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.3
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.2
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCPacking pack_ratio=0.1
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=1.0
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.9
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.8
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.7
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.6
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.5
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.4
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.3
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.2
CUDA_VISIBLE_DEVICES=2 python src/train.py experiment=malconv-bytes-train data_name=MFCAes pack_ratio=0.1
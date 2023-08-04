#!/bin/bash
# unseen
# CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=mlp-ember-test data_name=MFCUnseen
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=mlp-ember-test data_name=MFCEvolving
CUDA_VISIBLE_DEVICES=0 python src/eval.py experiment=mlp-ember-test data_name=MFCPacking pack_ratio=1.0
#!/bin/bash

# Loading the required module
# source /etc/profile
# module load anaconda/2021a

unset RANK

export PYTHONNOUSERSITE=True    # prevent using packages from base
# source activate th102_cu113_tgconda

python main_qm9.py \
    --output-dir 'models/qm9/gn_oc_small/target@1/' \
    --model-name 'gemnet_oc_small_qm9' \
    --target 1 \
    --data-path 'datasets/qm9' \
    --feature-type 'one_hot' \
    --weight-decay 0.01 \
    --sched plateau \
    --patience-epochs 3 \
    --lr 1e-4 \
    --min-lr 1e-6 \
    --no-amp \
    --no-model-ema \
    # --model-ema-decay 0.99

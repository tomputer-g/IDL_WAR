#!/bin/bash

#mkdir -p ./checkpoints/classifiers/

  python train_classifier.py \
    --wm_dir /DATA/CMU/surrogate_model/tree_ring \
    --org_dir /DATA/CMU/surrogate_model/unwatermarked \
    --out_dir checkpoints/myclassifiers/treeRing_classifier.pt \
    --data_cnt 7500 \
    --epochs 10 \

  
  python adv_attack.py \
    --wm_method treeRing \
    --wm_dir /DATA/CMU/surrogate_model/tree_ring \
    --org_dir /DATA/CMU/surrogate_model/unwatermarked \
    --model_dir checkpoints/myclassifiers/treeRing_classifier.pt \

#   python train_classifier.py \
#     --wm_dir /DATA/CMU/surrogate_model/stegastamp \
#     --org_dir /DATA/CMU/surrogate_model/unwatermarked \
#     --out_dir checkpoints/myclassifiers/stegaStamp_classifier.pt \
#     --data_cnt 7500 \
#     --epochs 10 \
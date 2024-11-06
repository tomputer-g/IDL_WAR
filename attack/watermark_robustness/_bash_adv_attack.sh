#!/bin/bash


  python adv_attack.py \
    --wm_method treeRing \
    --wm_dir /DATA/CMU/competition/adv_wm_images_stegaStamp_12 \
    --model_dir checkpoints/myclassifiers/treeRing_classifier.pt \

    

#   python adv_attack.py \
#     --wm_method stegaStamp \
#     --wm_dir /DATA/CMU/competition/Black/Black \
#     --model_dir checkpoints/classifiers/stegaStamp_classifier.pt \


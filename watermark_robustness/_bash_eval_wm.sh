#!/bin/bash


# python evaluate_watermark.py \
# --wm-method MBRS \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/MBRS \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \

 python evaluate_watermark.py \
  --wm_method rivaGan \
  --attack diffpure \
  --dataset competition \
  --data_dir /DATA/CMU/competition/Beige/Beige \
  --out_fname diffpure \
  --save_images

# python evaluate_watermark.py \
# --wm-method dwtDct \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/dwtDct \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \

# python evaluate_watermark.py \
# --wm-method dwtDctSvd \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/dwtDctSvd \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \

# python evaluate_watermark.py \
# --wm-method treeRing \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/treeRing \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \

# python evaluate_watermark.py \
# --wm-method watermarkDM \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/watermarkDM \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \

# python evaluate_watermark.py \
# --wm-method rivaGan \
# --attack diffpure \
# --dataset imagenet \
# --data-dir images/imagenet/rivaGan \
# --org-data-dir images/imagenet/org \
# --out-fname diffpure \


################  for adversarial attack

# example for eps=12

#   python evaluate_watermark.py \
#     --wm_method stegaStamp \
#     --attack no_aug \
#     --dataset imagenet \
#     --data_dir /DATA/CMU/surrogate_model/adv_images/adv_wm_images_stegaStamp_12 \
#     --org_data_dir /DATA/CMU/surrogate_model/unwatermarked \
#     --out_fname adv \

#   python evaluate_watermark.py \
#     --wm_method treeRing \
#     --attack no_aug \
#     --dataset imagenet \
#     --data_dir /DATA/CMU/surrogate_model/adv_images/wm/adv_wm_images_treeRing_12 \
#     --org_data_dir /DATA/CMU/surrogate_model/unwatermarked \
#     --out_fname adv \


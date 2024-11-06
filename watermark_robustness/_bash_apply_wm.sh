# CUDA_VISIBLE_DEVICES=6 \
#  python apply_watermark.py \
#    --wm_method treeRing \
#    --dataset imagenet \
#    --data_cnt 7500 \
#    --data_dir /DATA/CMU/surrogate_model/unwatermarked \
#    --out_dir /DATA/CMU/surrogate_model/tree_ring

# CUDA_VISIBLE_DEVICES=5 \
#  python apply_watermark.py \
#    --wm_method stegaStamp \
#    --dataset imagenet \
#    --data_cnt 7500 \
#    --data_dir /DATA/CMU/surrogate_model/unwatermarked \
#    --out_dir /DATA/CMU/surrogate_model/stegastamp

CUDA_VISIBLE_DEVICES=6 \
 python apply_watermark.py \
   --wm_method stegaStamp \
   --dataset competition \
   --data_cnt 300 \
   --data_dir /DATA/CMU/competition/Beige/beigebox \
   --out_dir /DATA/CMU/competition/Beige/adv_attacked



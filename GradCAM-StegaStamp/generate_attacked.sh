# for i in $(seq 0 25 75);
# do
#     for j in 5 11 31;
#     do
#         echo $i $j
#         python attack_gradcam.py ../../StegaStamp/saved_models/test/ --wm_images_dir ../../StegaStamp/data/watermarked_Stega\!\!/ --gradcam_results_dir gradcams/ --output_dir gradcam/gradcam --percentile_threshold $i --blur_size $j
#     done
# done

# for i in $(seq 0 25 75);
# do
#     for j in 5 11 31;
#     do
#         echo $i $j
#         python attack_randomized.py --wm_images_dir ../../StegaStamp/data/watermarked_Stega\!\!/ --output_dir randomized_attacked/randomized_attacked --percentile_threshold $i --blur_size $j
#     done
# done

for i in $(seq 0 25 75)
do
    for j in 5 11 31
    do
        echo $i $j
        python attack_residuals.py --wm_images_dir ../../StegaStamp/data/watermarked_Stega\!\!/ --unwm_images_dir ../../StegaStamp/data/unwatermarked --output_dir residuals_attacked/residuals_attacked --percentile_threshold $i --blur_size $j
    done
done

for i in $(seq 0 25 75)
do
    for j in 5 11 31
    do
        echo $i $j
        python attack_residuals.py --wm_images_for_residuals_dir ../../StegaStamp/data/watermarked_Hello --wm_images_dir ../../StegaStamp/data/watermarked_Stega\!\!/ --unwm_images_dir ../../StegaStamp/data/unwatermarked --output_dir residuals_attacked_diff_message/residuals_attacked_diff_message --percentile_threshold $i --blur_size $j
    done
done
attack="$1"

if [ "$attack" = "gradcam" ];
then
    for i in $(seq 0 25 75);
    do
        for j in 5 11 31;
        do
            # $2 -- path to saved stegastamp model
            echo $i $j
            python attack_gradcam.py $2 \
                                --wm_images_dir $3 \
                                --gradcam_results_dir $4 \
                                --output_dir gradcam_attacked/gradcam_attacked \
                                --percentile_threshold $i \
                                --blur_size $j
        done
    done
fi

if [ "$attack" = "randomized" ];
then
    for i in $(seq 0 25 75);
    do
        for j in 5 11 31;
        do
            echo $i $j
            python attack_randomized.py --wm_images_dir $2 \
                                        --gradcams_dir $3 \
                                        --output_dir randomized_attacked/randomized_attacked \
                                        --percentile_threshold $i \
                                        --blur_size $j
        done
    done
fi

if [ "$attack" = "residuals" ];
then
    for i in $(seq 0 25 75)
    do
        for j in 5 11 31
        do
            echo $i $j
            python attack_residuals.py --wm_images_dir $2 \
                                       --unwm_images_dir $3 \
                                       --output_dir residuals_attacked/residuals_attacked \
                                       --percentile_threshold $i \
                                       --blur_size $j
        done
    done
fi

if [ "$attack" = "residuals_diff_message" ];
then
    for i in $(seq 0 25 75)
    do
        for j in 5 11 31
        do
            echo $i $j
            python attack_residuals.py --wm_images_dir $2 \
                                       --wm_images_for_residuals_dir $3 \
                                       --unwm_images_dir $4 \
                                       --output_dir residuals_attacked_diff_message/residuals_attacked_diff_message \
                                       --percentile_threshold $i \
                                       --blur_size $j
        done
    done
fi

if [ "$attack" = "blur" ];
then
    for j in 5 11 31
    do
        echo $j
        python attack_blur.py --wm_images_dir $2 \
                              --output_dir $3 \
                              --blur_size $j
    done
fi
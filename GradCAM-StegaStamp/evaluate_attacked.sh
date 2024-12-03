for i in $(seq 0 25 75);
do
    for j in 5 11 31;
    do
        echo $i $j
        python calculate_accuracy.py ../../StegaStamp/saved_models/test/ \
            --images_dir gradcam_attacked/gradcam_${i}_${j}/ 2>/dev/null | grep "Detection Rate"
    done
done
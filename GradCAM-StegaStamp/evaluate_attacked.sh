for i in $(seq 0 25 75);
do
    for j in 5 11 31;
    do
        echo $i $j;
        python calculate_accuracy.py ../../StegaStamp/saved_models/test/ \
            --images_dir randomized_attacked/randomized_attacked_${i}_${j}/ 2>/dev/null | grep "Detection Rate";
        python -m pytorch_fid randomized_attacked/randomized_attacked_${i}_${j}/ ../../StegaStamp/data/unwatermarked | grep "FID";
    done
done
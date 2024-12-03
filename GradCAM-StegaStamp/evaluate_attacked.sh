stegastamp_model = $1
attacked_dir = $2
attacked_subfolder_name = $3
unwatermarked_dir = $4

for i in $(seq 0 25 75);
do
    for j in 5 11 31;
    do
        echo $i $j;
        python calculate_accuracy.py $stegastamp_model \
            --images_dir ${attacked_dir}/${attacked_subfolder_name}_${i}_${j}/ 2>/dev/null | grep "Detection Rate";
        python -m pytorch_fid ${attacked_dir}/${attacked_subfolder_name}_${i}_${j}/ $unwatermarked_dir | grep "FID";
    done
done
p_vals = []
    true_labels = []
    true_positive = 0
    false_positive = 0
    for i, watermarked, unwatermarked in zip(
        range(len(watermarked_images)), watermarked_images, unwatermarked_images
    ):
        unwatermarked_det = generator.detect(
            [unwatermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )
        watermarked_det = generator.detect(
            [watermarked], keys[i : i + 1], masks[i : i + 1], p_val_thresh=0.01
        )

        p_vals.append(1 - unwatermarked_det[0][0])
        # print(unwatermarked_det[0][0])
        true_labels.append(0)

        false_positive += 1 if unwatermarked_det[0][1] else 0

        p_vals.append(1 - watermarked_det[0][0])
        true_labels.append(1)

        true_positive += 1 if watermarked_det[0][1] else 0

    print(f"AUC: {roc_auc_score(true_labels, p_vals)}")
    print(f"TPR: {true_positive / (true_positive + false_positive)}")
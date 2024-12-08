import os

def main(tree_ring_eval_file, stegastamp_eval_file, p_val_threshold=0.01):
    tree_ring_results = {}
    with open(tree_ring_eval_file) as f:
        for line in f:
            filename, type_of_image, _, p_value = line.strip().split(",")
            if type_of_image == "unwatermarked":
                continue
            filename = os.path.basename(filename)
            p_value = float(p_value)
            tree_ring_results[filename] = (p_value < p_val_threshold)

    stegastamp_results = {}
    with open(stegastamp_eval_file) as f:
        for line in f:
            filename, detected = line.strip().split(",")
            filename = os.path.basename(filename)
            stegastamp_results[filename] = (detected=="true")

    total = 0
    detected = 0
    for filename in tree_ring_results.keys():
        total += 1

        if tree_ring_results[filename] or stegastamp_results[filename]:
            detected += 1

    print("Detection rate:", detected / total)



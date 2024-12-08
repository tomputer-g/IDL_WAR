import numpy as np
from sklearn.metrics import auc, roc_curve
import click

@click.command()
@click.option(
    "--eval_file",
    default="eval_probs.csv",
    show_default=True,
    help="Path to evaluation file generated by eval_tree_ring.py.",
)
def main(eval_file):
    true_labels = []
    probs = []
    with open(eval_file) as f:
        for line in f.readlines():
            data = line.strip().split(",")
            true_labels.append(int(data[2]))
            probs.append(-float(data[3]))

    fpr, tpr, thresholds = roc_curve(true_labels, probs)
    print("AUC:", auc(fpr, tpr))
    print("TPR@1%FPR:", tpr[np.where(fpr<0.01)[0][-1]])
    print("TPR@0.1%FPR:", tpr[np.where(fpr<0.001)[0][-1]])

if __name__ == "__main__":
    main()
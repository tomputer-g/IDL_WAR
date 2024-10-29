import numpy as np
from sklearn.metrics import auc, roc_curve

true_labels = []
probs = []
with open("eval_probs.csv") as f:
    for line in f.readlines():
        data = line.strip().split(",")
        true_labels.append(int(data[2]))
        probs.append(1-float(data[3]))
        
fpr, tpr, thresholds = roc_curve(true_labels, probs)
print(auc(fpr, tpr))
print(tpr[np.where(fpr<0.01)[0][-1]])
print(tpr[np.where(fpr<0.001)[0][-1]])
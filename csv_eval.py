import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

def BalAcc(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    TN = sum(~true_val & ~pred_val)
    FN = sum(true_val & ~pred_val)
    FP = sum(~true_val & pred_val)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)

    return (Sensitivity + Specificity)/2


def F1_score(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    # TN = sum((~true_val - min(~true_val)) & (~pred_val - min(~pred_val)))
    FN = sum(true_val & ~pred_val)
    FP = sum(~true_val & pred_val)

    if TP:
        return (2 * TP) / (2 * TP + FP + FN)
    else:
        return 0


def acc_score(true_val, pred_val, threshold=0.5):
    true_val, pred_val = true_val > threshold, pred_val > threshold
    TP = sum(true_val & pred_val)
    TN = sum(~true_val & ~pred_val)
    # FN = sum(true_val & ~pred_val)
    # FP = sum(~true_val & pred_val)

    if TP or TN:
        return (TP + TN) / len(true_val)
    else:
        return 0


if __name__ == "__main__":
    data = pd.read_csv("output.csv")
    threshold = 40

    print(f"Accuracy: {acc_score(data.y*100, data.y_est, threshold)*100}%")
    print(f"F1 score: {F1_score(data.y*100, data.y_est, threshold)}")
    print(f"Balanced accuracy: {BalAcc(data.y*100, data.y_est, threshold)}")


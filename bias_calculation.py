import numpy as np


def demographic_parity(y_pred, sensitive_attr):
    """
    dpd = |P(pred=1 | group_A) - P(pred=1 | group_B)|
    """
    pred_0 = y_pred[sensitive_attr == 0]
    pred_1 = y_pred[sensitive_attr == 1]
    
    prob_0 = pred_0.mean() if len(pred_0) > 0 else 0
    prob_1 = pred_1.mean() if len(pred_1) > 0 else 0
    
    dpd = abs(prob_0 - prob_1)
    
    return dpd, prob_0, prob_1


def equalized_odds(y_pred, y_true, sensitive_attr):
    """
    eod = max(|TPR_A - TPR_B|, |FPR_A - FPR_B|)
    """
    def get_rates(preds, labels):
        #  TP / (TP + FN)
        tp = ((preds == 1) & (labels == 1)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        #  FP / (FP + TN)
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        return tpr, fpr

    tpr_0, fpr_0 = get_rates(y_pred[sensitive_attr == 0], y_true[sensitive_attr == 0])
    tpr_1, fpr_1 = get_rates(y_pred[sensitive_attr == 1], y_true[sensitive_attr == 1])

    tpr_diff = abs(tpr_0 - tpr_1)
    fpr_diff = abs(fpr_0 - fpr_1)
    
    eod = max(tpr_diff, fpr_diff)
    
    return eod, tpr_diff, fpr_diff
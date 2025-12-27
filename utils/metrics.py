import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


def ppv_score(row, P, N): 
    """Calculate positive predictive value (PPV)"""
    denominator = (row['tpr']*P) + (row['fpr']*N)
    if denominator == 0:
        return np.nan
    ppv = (row['tpr']*P) / denominator
    return ppv


def npv_score(row, P, N): 
    """Calculate negative predictive value (NPV)"""
    denominator = (row['tnr']*N) + (row['fnr']*P)
    if denominator == 0:
        return np.nan
    npv = (row['tnr']*N) / denominator
    return npv


def sens_ppv_score(y_test, probas, metrics='both'):
    """Calculate sensitivity, ppv, npv at specific specificity levels"""
    N = np.bincount(y_test)[0]
    P = np.bincount(y_test)[1]
    fpr, tpr, thresholds = roc_curve(y_test, probas)
    tnr = 1-fpr
    fnr = 1-tpr
    ft_thres = pd.DataFrame(data=list(zip(fpr, tpr, tnr, fnr, thresholds)), 
                                columns=['fpr','tpr', 'tnr', 'fnr', 'thres'])

    if metrics == 'sensitivity':
        return (ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].tpr,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].tpr,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].tpr)
    elif metrics == 'ppv':
        ft_thres['ppv'] = ft_thres.apply(lambda row: ppv_score(row, P, N), axis=1)
        return (ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].ppv,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].ppv,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].ppv)
    elif metrics == 'npv':
        ft_thres['npv'] = ft_thres.apply(lambda row: npv_score(row, P, N), axis=1)
        return (ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].npv,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].npv,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].npv)
    elif metrics == 'both':
        ft_thres['ppv'] = ft_thres.apply(lambda row: ppv_score(row, P, N), axis=1)
        ft_thres['npv'] = ft_thres.apply(lambda row: npv_score(row, P, N), axis=1)
        return (ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].tpr,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].tpr,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].tpr,
                ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].ppv,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].ppv,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].ppv,
                ft_thres[ft_thres.fpr > 0.199999].reset_index().iloc[0].npv,
                ft_thres[ft_thres.fpr > 0.099999].reset_index().iloc[0].npv,
                ft_thres[ft_thres.fpr > 0.049999].reset_index().iloc[0].npv)
    

def show_results(test_metrics):
    """Print validation / test set results"""
    yte = test_metrics['true_y'].cpu().numpy().astype(int)
    yte_pred = test_metrics['pred_y'].cpu().numpy()

    prevalence = len([y for y in yte if y==1])/len(yte)
    print(f"Prevalence : {prevalence}")

    sens_80, sens_90, sens_95, ppv_80, ppv_90, ppv_95, npv_80, npv_90, npv_95 = sens_ppv_score(yte, yte_pred, 'both')
    print(f"NPV @Specificity-0.80 : {npv_80}")
    print(f"NPV @Specificity-0.90 : {npv_90}")
    print(f"NPV @Specificity-0.95 : {npv_95}")
    print()
    print(f"PPV @Specificity-0.80 : {ppv_80}")
    print(f"PPV @Specificity-0.90 : {ppv_90}")
    print(f"PPV @Specificity-0.95 : {ppv_95}")
    print()
    print(f"Sensitivity @Specificity-0.80 : {sens_80}")
    print(f"Sensitivity @Specificity-0.90 : {sens_90}")
    print(f"Sensitivity @Specificity-0.95 : {sens_95}")
    print()
    print(f"Relative risk-0.80 : {ppv_80/prevalence}") 
    print(f"Relative risk-0.90 : {ppv_90/prevalence}") 
    print(f"Relative risk-0.95 : {ppv_95/prevalence}") 
    print()
    print(f"Overall AUROC : {roc_auc_score(yte, yte_pred)}")


def adjusted_ppv(sensitivity_at_k, k, true_prevalence):
    """
    Estimate PPV at top k% under true prevalence,
    assuming model ranking is preserved.
    """
    # This is approximate and assumes ranking stability
    tp_rate = sensitivity_at_k * true_prevalence
    fp_rate = (k - sensitivity_at_k * true_prevalence)  # rough approximation
    return tp_rate / k



"""Stats & metrics helpers."""
import numpy as np
from scipy.stats import mannwhitneyu, shapiro
from sklearn.metrics import roc_auc_score, confusion_matrix

def stat_tests(scores_real: np.ndarray, scores_ai: np.ndarray):
    _, p_r = shapiro(scores_real)
    _, p_a = shapiro(scores_ai)
    U, p = mannwhitneyu(scores_real, scores_ai)
    return {"shapiro_real_p": p_r, "shapiro_ai_p": p_a, "mannwhitney_U": U, "p_value": p}

def auc(y_true, scores): return roc_auc_score(y_true, scores)
def cm(y_true, y_pred):  return confusion_matrix(y_true, y_pred)

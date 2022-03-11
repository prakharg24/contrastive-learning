import numpy as np
from sklearn import metrics

def sinedistance_eigenvectors(ustar, w):
    upredicted, _, _ = np.linalg.svd(w.T, full_matrices=False)
    sinedistance = np.linalg.norm(np.matmul(ustar, ustar.T) - np.matmul(upredicted, upredicted.T))/(2**0.5)
    return sinedistance


def classification_score(y_true, y_pred, mode='f1'):
    if mode=='f1':
        return metrics.f1_score(y_true, y_pred, average='macro')
    elif mode=='acc':
        return metrics.accuracy_score(y_true, y_pred)

def regression_score(y_true, y_pred, mode='rmse'):
    if mode=='rmse':
        return metrics.mean_squared_error(y_true, y_pred, squared=False)

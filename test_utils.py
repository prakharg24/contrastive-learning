import numpy as np
from sklearn import metrics
from sklearn import svm, linear_model

downstream_task_name = {'cls': 'Classification', 'reg': 'Regression'}

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

def downstream_score(dwn_mode, dwn_model,
                     representations_train, y_train,
                     representations_test, y_test):
    if dwn_mode=='cls':
        dwn_model = str(dwn_model)
        if dwn_model=='svm':
            predicted_labels = svm_classifier(representations_train, y_train, representations_test)
        else:
            raise Exception("Classification Model specified is not Implemented")

        final_score = classification_score(y_test, predicted_labels, mode='acc')

    elif dwn_mode=='reg':
        if dwn_model=='linear':
            predicted_labels = linear_regressor(representations_train, y_train, representations_test)
        else:
            raise Exception("Regression Model specified is not Implemented")

        final_score = regression_score(y_test, predicted_labels, mode='rmse')

    print("%s Task Score : %f" % (downstream_task_name[dwn_mode], final_score))
    return final_score

## Downstream Models
def svm_classifier(X_train, y_train, X_test):
    clf = svm.SVC()
    downstream_model = clf.fit(X_train, y_train)
    predicted_labels = clf.predict(X_test)
    return predicted_labels

def linear_regressor(X_train, y_train, X_test):
    reg = linear_model.LinearRegression()
    downstream_model = reg.fit(X_train, y_train)
    predicted_labels = reg.predict(X_test)
    return predicted_labels

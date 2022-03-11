from sklearn import svm, linear_model

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

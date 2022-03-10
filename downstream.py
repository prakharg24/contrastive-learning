from sklearn import svm

def svm_classifier(X_train, y_train, X_test):
    clf = svm.SVC()
    downstream_model = clf.fit(X_train, y_train)
    predicted_labels = clf.predict(X_test)
    return predicted_labels

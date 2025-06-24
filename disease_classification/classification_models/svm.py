from sklearn.svm import SVC
import numpy as np

from helpers import features_and_labels, print_classif_report

def linear_svm():
    X_test, X_train, y_test, y_train = features_and_labels()
    svclassifier = SVC(kernel='linear', degree=1, C=20)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)
    print_classif_report(y_test, y_pred)
    


def polynomial_svm():
    X_test, X_train, y_test, y_train = features_and_labels()
    svclassifier = SVC(kernel='poly', degree=1, C=20)
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)
    print(y_pred)
    print_classif_report(y_test, y_pred)
    


def rbf_svm():
    X_test, X_train, y_test, y_train = features_and_labels()
    svclassifier = SVC(kernel='rbf')
    svclassifier.fit(X_train, y_train)

    y_pred = svclassifier.predict(X_test)
    print_classif_report(y_test, y_pred)
    

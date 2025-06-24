from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt

from helpers import features_and_labels, print_classif_report

def knn_classify():
    X_test, X_train, y_test, y_train = features_and_labels()
    knn = KNeighborsClassifier(n_neighbors=3)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print_classif_report(y_test, y_pred)

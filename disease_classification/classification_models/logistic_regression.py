# Modeling the data as is
# Train model
from sklearn.linear_model import LogisticRegression
import numpy as np

from helpers import features_and_labels, print_classif_report

def lr_classify():
    X_test, X_train, y_test, y_train = features_and_labels()
    lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

    # Predict on training set
    lr_pred = lr.predict(X_test)

    # Checking accuracy
    print_classif_report(y_test, lr_pred)

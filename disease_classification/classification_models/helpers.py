import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from sklearn import preprocessing


def features_and_labels():
    speech_df = pd.read_csv("cleaned_speech.csv")
    speech_label_df = speech_df['753']
    speech_feature_df = speech_df.drop(['Unnamed: 0', '753'], axis=1)

    x_test, x_train, y_test, y_train = train_test_split(speech_label_df, speech_feature_df, test_size = 0.2)
    x_train, x_test = preprocess(x_train, x_test)

    return x_test, x_train, y_test, y_train

def train_test_split(speech_label_df, speech_feature_df, test_size):
    X_train, X_test, y_train, y_test = train_test_split(speech_feature_df, speech_label_df, test_size=test_size)
    return X_test, X_train, y_test, y_train

def preprocess(x_train, x_test):
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train = min_max_scaler.fit_transform(x_train)
    x_test = min_max_scaler.transform(x_test)

    return x_train, x_test


def print_classif_report(y_test, y_pred):
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


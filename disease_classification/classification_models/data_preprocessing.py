import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools


from sklearn import datasets, metrics
from sklearn.metrics import confusion_matrix


from sklearn.neighbors import KernelDensity
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split


from sklearn import preprocessing


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification


from sklearn.decomposition import FastICA
from sklearn.metrics import accuracy_score, log_loss
import sklearn.metrics as metrics
from sklearn.metrics import classification_report


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE



df=pd.read_csv('pd_speech_features.csv')

#data information 

print('info:',df.info())
print('head:',df.head())
print('shape:', df.shape)
print('description:', df.describe())

# Find missing values in the DataFrame
missing_values = df.isna()


# Locate columns with missing values
columns_with_missing = missing_values.any(axis=0)


# Get names of columns with missing values
columns_names = columns_with_missing[columns_with_missing].index.tolist()


# Print columns with missing values
print("Columns with missing values:")
print(columns_names)

df.class_pd.value_counts()   #how many unique rows   #gives unbalanced data


#extract features
feature_df= df.drop(['class_pd','id'], axis=1)
labels_df = df.class_pd #.unique()
feature_df.head()


#balance the data
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)

unique, counts = np.unique(transformed_label_df, return_counts=True)
print(np.asarray((unique, counts)).T)

transformed_label_df=transformed_label_df.values.reshape(-1,1)

a_final=np.concatenate((transformed_feature_df,transformed_label_df),axis=1)


df_new=pd.DataFrame(a_final)

#save the preprocesed data as csv
df_new.to_csv("cleaned_speech.csv")

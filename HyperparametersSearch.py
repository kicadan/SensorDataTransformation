import os
import csv
import random
import metrics
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics


pathDatasetFile = "D:\\Daniel\\Documents\\magister\\eksperymenty\\data\\cross_validation_all_features.csv"
columns = ["FE1W", "FE1P", "FE2W", "FE2P", "FE3W", "FE3P", "FE4W", "FE4P", "FE7XW", "FE7XP", "FE7YW", "FE7YP", "FE7ZW", "FE7ZP", "FE8XW", "FE8XP", "FE8YW", "FE8YP", "FE8ZW", "FE8ZP", "FE9XW", "FE9XP", "FE9YW", "FE9YP", "FE9ZW", "FE9ZP",
                     "FE10XW", "FE10XP", "FE10YW", "FE10YP", "FE10ZW", "FE10ZP", "FE11XW", "FE11XP", "FE11YW", "FE11YP", "FE11ZW", "FE11ZP", "F12XW", "F12XP", "F12YW", "F12YP", "F12ZW", "F12ZP", "F5W", "F5P", "F6W", "F6P"]


#LOADING DATA
data = []
with open(pathDatasetFile, 'r', newline='') as datasetfile:
    reader = csv.reader(datasetfile, delimiter=';')
    next(reader, None)
    for row in reader:
        data.append(row)
data = np.array(data)


results = []


#SVM

kernel = ['linear', 'poly', 'rbf', 'sigmoid']
probability = [1, 0] #true, false
C = [0.25, 0.5, 1, 10, 100]

for ker in kernel:
    for prob in probability:
        for c in C:
            df = pd.DataFrame(data[:, :48],
                              columns=columns)  # change nr of columns in file
            y = data[:, 48]  # last column in file
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
            clf = LogisticRegression(solver=sol, max_iter=mi, C=c).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append("kernel " + ker + ", probability " + str(prob) + ", C " + str(c) + ", accuracy " + str(
                metrics.accuracy_score(y_test, y_pred, normalize=True)) + ", precision " + str(
                metrics.precision_score(y_test, y_pred, pos_label="1")))


for res in results:
    print(res)

#Logistic Regression
"""
solver = ['liblinear']
max_iter = [100, 200, 500, 1000]
C = [0.25, 0.5, 1, 10, 100]
for sol in solver:
    for mi in max_iter:
        for c in C:
            df = pd.DataFrame(data[:, :48],
                              columns=columns)  # change nr of columns in file
            y = data[:, 48]  # last column in file
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
            clf = LogisticRegression(solver=sol, max_iter=mi, C=c).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append("solver " + sol + ", max_iter " + str(mi) + ", C " + str(c) + ", accuracy " + str(metrics.accuracy_score(y_test, y_pred, normalize=True)) + ", precision " + str(metrics.precision_score(y_test, y_pred, pos_label="1")))

for res in results:
    print(res)
"""
#Ranfom Forest
"""
n_estimators = [10, 20, 50, 100, 200, 500, 1000]
criterion = ['gini', 'entropy']
max_features = ['auto', 'sqrt', 'log2']
min_samples_leaf = [ 1, 2, 3, 4, 5]

for ne in n_estimators:
    for crit in criterion:
        for mf in max_features:
            for msl in min_samples_leaf:
                df = pd.DataFrame(data[:, :48],
                                  columns=columns)  # change nr of columns in file
                y = data[:, 48]  # last column in file
                X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
                clf = RandomForestClassifier(n_estimators=ne, criterion=crit, max_features=mf, min_samples_leaf=msl).fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                results.append("criterion " + crit + ", n_estimators " + str(ne) + ", max_features " + mf + ", min_samples_leaf " + str(msl) + ", accuracy " + str(
                    metrics.accuracy_score(y_test, y_pred, normalize=True)) + ", precision " + str(
                    metrics.precision_score(y_test, y_pred, pos_label="1")))

for res in results:
    print(res)
"""

#Decision Tree

criterion = ['gini', 'entropy']
max_features = ['auto', 'sqrt', 'log2']
min_samples_leaf = [ 1, 2, 3, 4, 5]

for crit in criterion:
    for mf in max_features:
        for msl in min_samples_leaf:
            df = pd.DataFrame(data[:, :48],
                              columns=columns)  # change nr of columns in file
            y = data[:, 48]  # last column in file
            X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
            clf = DecisionTreeClassifier(criterion=crit, min_samples_leaf=msl, max_features=mf).fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            results.append("criterion " + crit + ", max_features " + mf + ", min_samples_leaf " + str(msl) + ", accuracy " + str(
                metrics.accuracy_score(y_test, y_pred, normalize=True)) + ", precision " + str(
                metrics.precision_score(y_test, y_pred, pos_label="1")))

for res in results:
    print(res)


def cross_validation(clf):
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    indexes = [ [0, 36], [37, 73], [74, 110], [111, 146], [147, 182], [183, 216], [217, 253],[254, 290], [291, 328], [329,365]] #indexes of specified groups

    for i in range(0,10):
        X_test = data[np.arange(indexes[i][0], indexes[i][1]+1), :48]
        X_test = X_test.astype(np.float64)
        X_train = np.delete(data, np.arange(indexes[i][0],indexes[i][1]+1), axis=0)[:, :48]
        X_train = X_train.astype(np.float64)
        y_test = data[np.arange(indexes[i][0], indexes[i][1]+1), 48]
        y_train = np.delete(data, np.arange(indexes[i][0],indexes[i][1]+1), axis=0)[:, 48]
        print(clf.score(X_test, y_test))


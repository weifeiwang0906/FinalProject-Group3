import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

ron_train = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/realornot_trainencode.csv')

labels_train = np.array(ron_train['768'])
features_train = ron_train.drop('768', axis=1)
features_list_train = list(features_train.columns)
features_train = np.array(features_train)

features_train_train, features_train_test, labels_train_train, labels_train_test = train_test_split(
        features_train, labels_train, test_size=0.2, random_state=100)
n_e = list()
accu = list()
for n_estimator in range(100, 3000, 100):
    rf = RandomForestClassifier(n_estimators=n_estimator, random_state=100)
    rf.fit(features_train_train, labels_train_train)
    pred_rf = rf.predict(features_train_test)
    a = accuracy_score(labels_train_test, pred_rf)
    n_e.append(n_estimator)
    accu.append(a)
    print('n_estimator:', n_estimator)
    print('Accuracy: %.3f' % accuracy_score(labels_train_test, pred_rf))

plt.plot(n_e, accu)
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.savefig('rf_para.png')
plt.show()





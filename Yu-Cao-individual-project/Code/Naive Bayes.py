import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

# 1. read data and see the information about them
ron_train = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/realornot_trainencode.csv')
print(ron_train.head())
print(ron_train.describe())

ron_test = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/realornot_testencode.csv')
print(ron_test.head())
print(ron_test.describe())

ori_test = pd.read_csv('/Users/apple/Documents/me/GWU/courses/ds6103/python03/Data-Mining-master/final project/nlp-getting-started/test.csv')
id = ori_test['id']

# 2. data pre-processing and feature engineering
ron_train.drop('Unnamed: 0', axis=1, inplace=True)
ron_test.drop('Unnamed: 0', axis=1, inplace=True)
# abels_train = np.array(ron_train['768'])
# features_train = ron_train.drop('768', axis=1)
labels_train = ron_train.iloc[:, -1]
features_train = ron_train.iloc[:, 0:-1]
features_list_train = list(features_train.columns)
features_train = np.array(features_train)

# 3. modeling
# 3.1 train and test
nb_all = GaussianNB()
nb_all.fit(features_train, labels_train)

pred_all_nb = nb_all.predict(ron_test)

with open('test_Naive Bayes.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['id', 'target'])

    for i in range(1, len(pred_all_nb)+1):
        writer.writerow([id[i-1], pred_all_nb[i-1]])

# 3.2 only train file. train set is splited into train and test
features_train_train, features_train_test, labels_train_train, labels_train_test = train_test_split(
    features_train, labels_train, test_size=0.2, random_state=100
)
nb = GaussianNB()
nb.fit(features_train_train, labels_train_train)

# 4. prediction (only train)
pred_nb = nb.predict(features_train_test)

# 5. output (only train)
# 5.1 Accuracy, confusion matrix and classification matrix
print('Accuracy: %.3f' % accuracy_score(labels_train_test, pred_nb))
print('Confusion matrix is:')
print(confusion_matrix(labels_train_test, pred_nb))
print('Classification:')
print(classification_report(labels_train_test, pred_nb))

# 5.2 ROC Curve
ns_probs_nb = [0 for _ in range(len(labels_train_test))]
nb_probs = nb.predict_proba(features_train_test)
nb_probs = nb_probs[:, 1]

ns_auc_nb = roc_auc_score(labels_train_test, ns_probs_nb)
nb_auc = roc_auc_score(labels_train_test, nb_probs)
print('No Skill: ROC AUC=%.3f' % ns_auc_nb)
print('Naive Bayes: ROC AUC=%.3f' % nb_auc)

ns_fpr_nb, ns_tpr_nb, _ = roc_curve(labels_train_test, ns_probs_nb)
nb_fpr, nb_tpr, _ = roc_curve(labels_train_test, nb_probs)

plt.plot(ns_fpr_nb, ns_tpr_nb, linestyle='--')
plt.plot(nb_fpr, nb_tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['No Skill', 'Naive Bayes'])
plt.savefig('nb_ROC.png')
plt.show()

# 5.3 P-R Curve
nb_probs_pr = nb.predict_proba(features_train_test)
nb_probs_pr = nb_probs_pr[:, 1]

nb_precision, nb_recall, _ = precision_recall_curve(labels_train_test, nb_probs_pr)
nb_f1, nb_auc = f1_score(labels_train_test, pred_nb), auc(nb_recall, nb_precision)
print('Naive Bayes: f1=%.3f, P-R AUC=%.3f' % (nb_f1, nb_auc))
ns_pr_nb = len(labels_train_test[labels_train_test==1])/len(labels_train_test)

plt.plot([0, 1], [ns_pr_nb, ns_pr_nb], linestyle='--')
plt.plot(nb_recall, nb_precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(['No Skill', 'Naive Bayes'])
plt.savefig('nb_PR.png')
plt.show()




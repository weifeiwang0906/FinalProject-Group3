import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pydot
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from sklearn.tree import export_graphviz

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
# labels_train = np.array(ron_train['768'])
# features_train = ron_train.drop('768', axis=1)
labels_train = ron_train.iloc[:, -1]
features_train = ron_train.iloc[:, 0:-1]
features_list_train = list(features_train.columns)
features_train = np.array(features_train)

# 3. modeling
# 3.1 train and test
rf_all = RandomForestClassifier(n_estimators=2000, random_state=100)
rf_all.fit(features_train, labels_train)

# prediction
pred_all_rf = rf_all.predict(ron_test)

# save the result into a csv file and submit it to Kaggle
with open('test_Random Forest.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    writer.writerow(['id', 'target'])

    for i in range(1, len(pred_all_rf)+1):
        writer.writerow([id[i-1], pred_all_rf[i-1]])

# 3.2 only train file. train set is split into train and test
features_train_train, features_train_test, labels_train_train, labels_train_test = train_test_split(
    features_train, labels_train, test_size=0.2, random_state=100
)
rf = RandomForestClassifier(n_estimators=1660, random_state=100)
rf.fit(features_train_train, labels_train_train)

# 4. prediction (only train)
pred_rf = rf.predict(features_train_test)

# 5. output (only train)
# 5.1 Accuracy, confusion matrix and classification matrix
print('Accuracy: %.3f' % accuracy_score(labels_train_test, pred_rf))
print('Confusion matrix is:')
print(confusion_matrix(labels_train_test, pred_rf))
print('Classification:')
print(classification_report(labels_train_test, pred_rf))

# 5.2 ROC Curve
ns_probs_rf = [0 for _ in range(len(labels_train_test))]
rf_probs = rf.predict_proba(features_train_test)
rf_probs = rf_probs[:, 1]

ns_auc_rf = roc_auc_score(labels_train_test, ns_probs_rf)
rf_auc = roc_auc_score(labels_train_test, rf_probs)
print('No Skill: ROC AUC=%.3f' % ns_auc_rf)
print('Random Forest: ROC AUC=%.3f' % rf_auc)

ns_fpr_rf, ns_tpr_rf, _ = roc_curve(labels_train_test, ns_probs_rf)
rf_fpr, rf_tpr, _ = roc_curve(labels_train_test, rf_probs)

plt.plot(ns_fpr_rf, ns_tpr_rf, linestyle='--')
plt.plot(rf_fpr, rf_tpr, marker='.')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(['No Skill', 'Random Forest'])
plt.savefig('rf_ROC.png')
plt.show()

# 5.3 P-R Curve
rf_probs_pr = rf.predict_proba(features_train_test)
rf_probs_pr = rf_probs_pr[:, 1]

rf_precision, rf_recall, _ = precision_recall_curve(labels_train_test, rf_probs_pr)
rf_f1, rf_auc = f1_score(labels_train_test, pred_rf), auc(rf_recall, rf_precision)
print('Random Forest: f1=%.3f, P-R AUC=%.3f' % (rf_f1, rf_auc))
ns_pr_rf = len(labels_train_test[labels_train_test==1])/len(labels_train_test)

plt.plot([0, 1], [ns_pr_rf, ns_pr_rf], linestyle='--')
plt.plot(rf_recall, rf_precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(['No Skill', 'Random Forest'])
plt.savefig('rf_PR.png')
plt.show()

# 5.4 A decision tree
tree = rf.estimators_[5]
export_graphviz(tree, out_file='tree.dot', feature_names=features_list_train,
                rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')

# 5.5 A smaller tree
rf_small = RandomForestClassifier(n_estimators=1660, max_depth=3)
rf_small.fit(features_train_train, labels_train_train)
tree_small = rf_small.estimators_[5]
export_graphviz(tree_small, out_file='small_tree.dot', feature_names=features_list_train,
                rounded=True, precision=1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png')




import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt

# 1. read data
train = pd.read_csv("realornot_trainencode.csv")
test = pd.read_csv("realornot_testencode.csv")
test_origin = pd.read_csv("test.csv")
test_id = test_origin["id"]
print(train.shape)

# 2. data pre-processing and feature engineering
train.drop("Unnamed: 0", axis=1, inplace=True)
test.drop("Unnamed: 0", axis=1, inplace=True)
train_x = train.iloc[:, 0:-1]
train_y = train.iloc[:, -1]

# 3. modeling
svm = SVC(C= 1.15,probability=True)
svm.fit(train_x, train_y)

# 4. prediction
predict1 = svm.predict(test)

# 5. output
result = pd.DataFrame()
result["id"] = test_id
result["target"] = predict1
result["id"] = result["id"].apply(lambda x: str(int(x)))
result["target"] = result["target"].apply(lambda x: str(int(x)))
result.to_csv("result_svm.csv", index=False)


# 6. Plot ROC an PR
# 6.1 ROC
train_y = pd.DataFrame(train_y)
x_train, x_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=22)
no_skill = len(y_test[y_test["768"] == 1])/ len(y_test)
model = SVC(C= 1.15,probability=True)
model.fit(x_train, y_train)
probs = model.predict_proba(x_test)
preds = probs[:, 1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='SVM AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# 6.2 PR
precision, recall, threshold_ = precision_recall_curve(y_test, preds)
plt.title('Precision Recall')
plt.plot(recall, precision, "b", label="SVM")
plt.plot([0, 1], [no_skill, no_skill], 'r--', label="No Skill")
plt.legend(loc='upper right')

plt.ylabel('Precision')
plt.xlabel('Recall')
plt.show()






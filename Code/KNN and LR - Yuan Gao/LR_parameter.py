from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings


def warn(*arg, **kwargs):
    pass


warnings.warn = warn

data = pd.read_csv("realornot_trainencode.csv")

data.drop("Unnamed: 0", axis=1, inplace=True)
train_x = data.iloc[:, 0:-1]
train_y = data.iloc[:, -1]
true_samples = len(train_y[train_y == 1])
false_samples = len(train_y[train_y == 0])
true_weight = true_samples/len(train_y)
false_weight = false_samples/len(train_y)
class_weight = {0: false_weight, 1: true_weight}
# parameters
# penalty - ["l1, l2"]
# solver - ["liblinear", "lbfgs", "newton-cg"]->l2
# ["sag"] large sample or l1
# max_iter default=100

iteration_num = [100, 150, 200, 250, 300]
solver = ["liblinear", "lbfgs", "newton-cg"]

kfold = KFold(n_splits=5, random_state=233)
fold_index = 1
result = pd.DataFrame()
index = 0
for train_index, test_index in kfold.split(data):
    train_x = data.iloc[train_index, :-1]
    train_y = data.iloc[train_index, -1]
    test_x = data.iloc[test_index, :-1]
    test_y = data.iloc[test_index, -1]
    for i in iteration_num:
        for s in solver:
            model = LogisticRegression(solver=s, max_iter=i)
            model.fit(train_x, train_y)
            prediction = model.predict(test_x)
            accuracy = accuracy_score(test_y, prediction)
            print("fold = {}, iteration num = {}, solver = {}, accuracy = {}".format(fold_index, i, s, accuracy))
            result.loc[index, "Fold"] = fold_index
            result.loc[index, "iter"] = i
            result.loc[index, "solver"] = s
            result.loc[index, "accuracy"] = accuracy
            index += 1
    fold_index += 1

for i in iteration_num:
    acc_avg = result[result["iter"]==i]["accuracy"].mean()
    print("iteration={}, average accuracy={}".format(i, acc_avg))

for s in solver:
    acc_avg = result[result["solver"]==s]["accuracy"].mean()
    print("solver={}, average accuracy={}".format(s, acc_avg))

result.to_csv("lr_parameters.csv")


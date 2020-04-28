from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd
import warnings


# def warn(*args, **kwargs):
#     pass
# warnings.warn = warn

data = pd.read_csv("realornot_trainencode.csv")

data.drop("Unnamed: 0", axis=1, inplace=True)
train_x = data.iloc[:, 0:-1]
train_y = data.iloc[:, -1]


max_depth = [100, 150, 200, 250, 300, None]
max_leaf_nodes = [100, 150, 200, 250, 300, None]

kfold = KFold(n_splits=5, random_state=233)
fold_index = 1
result = pd.DataFrame()
index = 0
for train_index, test_index in kfold.split(data):
    train_x = data.iloc[train_index, :-1]
    train_y = data.iloc[train_index, -1]
    test_x = data.iloc[test_index, :-1]
    test_y = data.iloc[test_index, -1]
    for d in max_depth:
        for s in max_leaf_nodes:
            model = DecisionTreeClassifier(max_depth=d, max_leaf_nodes=s)
            model.fit(train_x, train_y)
            prediction = model.predict(test_x)
            accuracy = accuracy_score(test_y, prediction)
            print("fold = {}, max_depth = {}, max_leaf_nodes = {}, accuracy = {}".format(fold_index, d, s, accuracy))
            result.loc[index, "Fold"] = fold_index
            result.loc[index, "max depth"] = d
            result.loc[index, "max leaf nodes"] = s
            result.loc[index, "accuracy"] = accuracy
            index += 1
    fold_index += 1

for d in max_depth:
    acc_avg = result[result["max depth"] == d]["accuracy"].mean()
    print("max depth={}, average accuracy={}".format(d, acc_avg))

for l in max_leaf_nodes:
    acc_avg = result[result["max leaf nodes"] == l]["accuracy"].mean()
    print("max leaf nodes={}, average accuracy={}".format(l, acc_avg))

result.to_csv("dt_parameters.csv")


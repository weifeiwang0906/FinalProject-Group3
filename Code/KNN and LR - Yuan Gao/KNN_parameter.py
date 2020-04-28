from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pandas as pd

data = pd.read_csv("realornot_trainencode.csv")

data.drop("Unnamed: 0", axis=1, inplace=True)

kfold = KFold(n_splits=5, random_state=111)
k_list = list(range(5, 20))

fold_num = 1
fold_result_list = []
best_k_list = []
for train_index, test_index in kfold.split(data):
    train_x = data.iloc[train_index, :-1]
    train_y = data.iloc[train_index, -1]
    test_x = data.iloc[test_index, :-1]
    test_y = data.iloc[test_index, -1]
    k_value_dict = {}
    for k in k_list:
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        model.fit(train_x, train_y)
        predict_y = model.predict(test_x)
        score = accuracy_score(test_y, predict_y)
        k_value_dict[k] = score
        print("fold = {}, k = {}, accuracy = {}".format(fold_num, k, score))
    best_k = max(k_value_dict, key=k_value_dict.get)
    best_k_list.append(best_k)
    fold_num += 1
    fold_result_list.append(k_value_dict)
print(best_k_list)
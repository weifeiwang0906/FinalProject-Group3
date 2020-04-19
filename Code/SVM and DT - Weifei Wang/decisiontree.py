import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC

# 1. read data
train = pd.read_csv("realornot_trainencode.csv")
test = pd.read_csv("realornot_testencode.csv")
test_origin = pd.read_csv("test.csv")
test_id = test_origin["id"]
# 2. data pre-processing and feature engineering
train.drop("Unnamed: 0", axis=1, inplace=True)
test.drop("Unnamed: 0", axis=1, inplace=True)
train_x = train.iloc[:, 0:-1]
train_y = train.iloc[:, -1]

# 3. modeling
model1 = DecisionTreeClassifier()
# model2 = SVC()
# model3 = LinearSVC()
model1.fit(train_x, train_y)
#model2.fit(train_x, train_y)
#model3.fit(train_x, train_y)

# 4. prediction
predict1 = model1.predict(test)

# 5. output
result = pd.DataFrame()
result["id"] = test_id
result["target"] = predict1
result["id"] = result["id"].apply(lambda x: str(int(x)))
result["target"] = result["target"].apply(lambda x: str(int(x)))
result.to_csv("result_dt.csv", index=False)
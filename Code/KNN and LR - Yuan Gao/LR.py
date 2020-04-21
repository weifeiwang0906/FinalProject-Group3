import pandas as pd
from sklearn.linear_model import LogisticRegression

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
LogisticR= LogisticRegression(max_iter=80, solver="lbfgs")
LogisticR.fit(train_x, train_y)


# 4. prediction
predict1 = LogisticR.predict(test)

# 5. output
result = pd.DataFrame()
result["id"] = test_id
result["target"] = predict1
result["id"] = result["id"].apply(lambda x: str(int(x)))
result["target"] = result["target"].apply(lambda x: str(int(x)))
result.to_csv("result_LR.csv", index=False)
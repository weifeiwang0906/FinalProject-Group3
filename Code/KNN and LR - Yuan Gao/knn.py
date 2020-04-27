import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
KNN= KNeighborsClassifier(n_neighbors=12,n_jobs=-1)
KNN.fit(train_x, train_y)


# 4. prediction
predict1 = KNN.predict(test)

# 5. output
result = pd.DataFrame()
result["id"] = test_id
result["target"] = predict1
result["id"] = result["id"].apply(lambda x: str(int(x)))
result["target"] = result["target"].apply(lambda x: str(int(x)))
result.to_csv("result_KNN.csv", index=False)
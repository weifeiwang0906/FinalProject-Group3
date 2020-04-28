from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd

data = pd.read_csv("realornot_trainencode.csv")

data.drop("Unnamed: 0", axis=1, inplace=True)
train_x = data.iloc[:, 0:-1]
train_y = data.iloc[:, -1]

c_list = [1.0, 1.15, 1.25, 1.35, 1.45]
gammas = [2.0, 2.15, 2.0, 2.25, 2.3, "scale"]
param_grid = {'C': c_list, 'gamma': gammas}
grid_search1 = GridSearchCV(SVC(), param_grid, cv=5, n_jobs=8)
grid_search1.fit(train_x, train_y)
print("best parameter for SVC", grid_search1.best_params_)
print("best score for SVC", grid_search1.best_score_)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


np.random.seed(2021)

water = pd.read_csv(os.getcwd() + "\\Seowoo\\part4\\ch06\\water_potability.csv")

# 열방향 axes = 0  행방향
data = water.drop(["Potability"], axis=1)
label = water["Potability"]

print(data)
print(data.describe())
print(data.isna().sum())

# 빈데이터 제거
# row 제거
print(data.isna().sum(axis=1))
na_cnt = data.isna().sum(axis=1)
print(na_cnt)
drop_idx = na_cnt.loc[na_cnt > 0].index
drop_idx
drop_row = data.drop(drop_idx, axis=0)
drop_row.shape
data.shape

# column 제거
na_cnt = data.isna().sum()
drop_cols = na_cnt.loc[na_cnt > 0].index
drop_cols
data = data.drop(drop_cols, axis=1)

from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_label)}, {len(train_label)/len(data):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(data):.2f}")

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
from sklearn.model_selection import GridSearchCV
params = {
    "n_neighbors": [i for i in range(1, 12, 2)],
    "p": [1, 2]
}
params
# n_jobs resource
grid_cv = GridSearchCV(knn, param_grid=params, cv=3, n_jobs=-1)
grid_cv.fit(train_data, train_label)

print(f"Best score of paramter search is: {grid_cv.best_score_:.4f}")
grid_cv.best_params_

print("Best parameter of best score is")
print(f"\t n_neighbors: {grid_cv.best_params_['n_neighbors']}")
print(f"\t p: {grid_cv.best_params_['p']}")

train_pred = grid_cv.best_estimator_.predict(train_data)
test_pred = grid_cv.best_estimator_.predict(test_data)
# 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_label, train_pred)
test_acc = accuracy_score(test_label, test_pred)

print(f"train accuracy is {train_acc:.4f}")
print(f"test accuracy is {test_acc:.4f}")
print(data)
# 변수들의 범위가 크기때문에 규모가 큰 것이 영향력도 크다
# 스케일링 필요

# 스케일링 후 예측
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

scaling_knn = KNeighborsClassifier()
scaling_grid_cv = GridSearchCV(scaling_knn, param_grid=params, n_jobs=-1)

scaling_grid_cv.fit(scaled_train_data, train_label)
scaling_grid_cv.best_score_
scaling_grid_cv.best_params_

# 평가
scaling_train_pred = scaling_grid_cv.best_estimator_.predict(scaled_train_data)
scaling_test_pred = scaling_grid_cv.best_estimator_.predict(scaled_test_data)

scaling_train_acc = accuracy_score(train_label, scaling_train_pred)
scaling_test_acc = accuracy_score(test_label, scaling_test_pred)

print(f"Scaled data train accuracy is {scaling_train_acc:.4f}")
print(f"Scaled data test accuracy is {scaling_test_acc:.4f}")

print(f"test accuracy is {test_acc:.4f}")
print(f"Scaled data test accuracy is {scaling_test_acc:.4f}")

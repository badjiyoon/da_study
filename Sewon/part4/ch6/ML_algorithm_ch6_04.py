#CH06_04. 음수 가능 여부 판단(python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
water = pd.read_csv("C:/Users/sewon/Documents/da_study/Sewon/part4/ch6/water_potability.csv")

data = water.drop(["Potability"], axis=1)
label = water["Potability"]

#1.2 Data EDA
"""
데이터의 변수들을 확인 
count를 확인하면 count들이 다른 것을 확인할 수 있음
"""
data.describe()
data.isna() #빈값 찾기
data.isna().sum()

#1.3 Data Preprocess
"""
빈 데이터를 제거하는 전처리
1) row를 제거하는 법
2) column을 제거하는 방법
"""

#1.3.1 row를 제거하는 방법
data.isna().sum(axis=1)

na_cnt = data.isna().sum(axis=1)
na_cnt

drop_idx = na_cnt.loc[na_cnt > 0].index
drop_idx

drop_row = data.drop(drop_idx, axis=0)
drop_row.shape
data.shape

#1.3.2 column을 제거하는 방법
na_cnt = data.isna().sum()
drop_cols = na_cnt.loc[na_cnt > 0].index

drop_cols

data = data.drop(drop_cols, axis=1)

#1.4 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_label)}, {len(train_label)/len(data):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(data):.2f}")

#2. KNN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

#2.1 Best Hyper Parameter
"""
KNeighborsClassifier에서 탐색해야 할 argument
-n_neighbors: 몇 개의 이웃으로 예측할 것 인지
-p: 거리를 어떤 방식으로 계산할지
 1) manhattan distance
 2) euclidean distance
"""
from sklearn.model_selection import GridSearchCV

#2.1.1 탐색 범위 선정
params = {
    "n_neighbors": [i for i in range(1, 12, 2)],
    "p": [1, 2]
}
params

#2.1.2 탐색
grid_cv = GridSearchCV(knn, param_grid=params, cv=3, n_jobs=-1)
grid_cv.fit(train_data, train_label)

#2.1.3 결과
print(f"Best score of paramter search is: {grid_cv.best_score_:.4f}")
grid_cv.best_params_

print("Best parameter of best score is")
print(f"\t n_neighbors: {grid_cv.best_params_['n_neighbors']}")
print(f"\t p: {grid_cv.best_params_['p']}")

#2.1.4 예측
train_pred = grid_cv.best_estimator_.predict(train_data)
test_pred = grid_cv.best_estimator_.predict(test_data)

#2.1.5 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_label, train_pred)
test_acc = accuracy_score(test_label, test_pred)

print(f"train accuracy is {train_acc:.4f}")
print(f"test accuracy is {test_acc:.4f}")

#3. Scaling을 할 경우
#3.1 Data Scaling
"""
KNN은 거리를 기반으로 하는 알고리즘이기 때문에 데이터의 크기에 영향을 받습니다.
Scaling을 진행해 크기를 맞춰줍니다.
"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

#3.2 탐색
scaling_knn = KNeighborsClassifier()
scaling_grid_cv = GridSearchCV(scaling_knn, param_grid=params, n_jobs=-1)

scaling_grid_cv.fit(scaled_train_data, train_label)

scaling_grid_cv.best_score_
scaling_grid_cv.best_params_

#3.3 평가
scaling_train_pred = scaling_grid_cv.best_estimator_.predict(scaled_train_data)
scaling_test_pred = scaling_grid_cv.best_estimator_.predict(scaled_test_data)

scaling_train_acc = accuracy_score(train_label, scaling_train_pred)
scaling_test_acc = accuracy_score(test_label, scaling_test_pred)

print(f"Scaled data train accuracy is {scaling_train_acc:.4f}")
print(f"Scaled data test accuracy is {scaling_test_acc:.4f}")

#4. 마무리
print(f"test accuracy is {test_acc:.4f}")
print(f"Scaled data test accuracy is {scaling_test_acc:.4f}")
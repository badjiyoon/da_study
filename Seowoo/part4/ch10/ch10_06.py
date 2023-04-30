# 샘플 데이터와 Stacking Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)
## 1. Data
### 1.1 Sample Data
from sklearn.datasets import make_regression

data, label = make_regression(n_samples=1000, n_features=20, n_informative=15, noise=0.1, random_state=1)
data.shape, label.shape


### 1.2 Data EDA
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca_data = pca.fit_transform(data)
plt.scatter(pca_data, label)

### 1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)

train_data

## 2. 개별 모델의 성능
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

models = {
    'knn': KNeighborsRegressor(),
    'tree': DecisionTreeRegressor(),
    'svm': SVR(),
}

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

names = []
results = []

for name, model in models.items():
    result = cross_val_score(model, train_data, train_label, cv=3, scoring="neg_mean_absolute_error")
    names += [name]
    results += [result]

names
results

import sklearn

sklearn.metrics.SCORERS.keys()

results

plt.figure(figsize=(8, 8))
plt.boxplot(results, labels=names)
plt.show()

for name, model in models.items():
    model.fit(train_data, train_label)
    test_pred = model.predict(test_data)
    acc = mean_absolute_error(test_label, test_pred)
    print(f"Model {name} test mean absoulte erorr is {acc:.4}")

## 3. Stacking
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

stacking = StackingRegressor(
    estimators=list(models.items()),
    final_estimator=LinearRegression(),
    cv=3
)

stacking_result = cross_val_score(stacking, train_data, train_label, cv=3, scoring="neg_mean_absolute_error")
stacking_result

all_result = []
all_result.extend(results)
all_result.append(stacking_result)

plt.figure(figsize=(8, 8))
plt.boxplot(all_result, labels=names + ["stacking"])
plt.show()

for name, model in models.items():
    test_pred = model.predict(test_data)
    acc = mean_absolute_error(test_label, test_pred)
    print(f"Model {name} test mean absoulte erorr is {acc:.4}")

stacking.fit(train_data, train_label)
stacking_pred = stacking.predict(test_data)
stacking_mae = mean_absolute_error(test_label, stacking_pred)

print(f"Model stacking mean absoulte erorr is {stacking_mae:.4}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
np.random.seed(2021)

# 1. Data
# 1.1 Data Load
housing = fetch_california_housing()
data, target = housing['data'], housing['target']
# 데이터 줄이기
print(f'data.shape: {data.shape}')
data = data[:2000]
target = target[:2000]

# 1.2 Data EDA
describe = pd.DataFrame(data, columns=housing['feature_names']).describe()
print(f'data describe: {describe}')
describe1 = pd.Series(target).describe()
print(f'target data describe: {describe1}')
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 10))
for i, feature_name in enumerate(housing["feature_names"]):
    ax = axes[i // 4, i % 4]
    ax.scatter(data[:, i], target)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("price")

plt.show()

# 1.3 Data split
train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)
# 2. Random Forest
rf_regressor = RandomForestRegressor()
# 2.1 학습
rf_regressor.fit(train_data, train_target)
# 2.2 예측
train_pred = rf_regressor.predict(train_data)
test_pred = rf_regressor.predict(test_data)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].scatter(train_target, train_pred)
axes[0].set_xlabel("predict")
axes[0].set_ylabel("real")

axes[1].scatter(test_target, test_pred)
axes[1].set_xlabel("predict")
axes[1].set_ylabel("real")

plt.show()

# 2.3 평가
train_mse = mean_squared_error(train_target, train_pred)
test_mse = mean_squared_error(test_target, test_pred)

print(f'train mean squared error is {train_mse: .4f}')
print(f'test mean squared error is {test_mse: .4f}')

# 2.4 Feature Importance
feature_importance = pd.Series(rf_regressor.feature_importances_, index=housing['feature_names'])
feature_importance.sort_values(ascending=True).plot(kind='barh')

# 3. Best Hyper Prameter
# RandomForestClassifier에서 주로 탐색하는 Argument
# n_estimators : 몇개의 나무를 생성할 것 인지 정합니다.
# criterion : 어떤 정보 이득을 기준으로 데이터를 나눌지 정합니다. 'gini', 'entropy'
# max_depth : 나무의 최대 길이를 정합니다.
# min_samples_split : 노드가 나눠질 수 있는 최소 데이터 개수를 정합니다.
# 탐색해야할 Argument들이 많을 때 일일이 지정하거나 for loop를 작성하기 힘들어진다.
# 이 때 사용할 수 있는 것이 skearn.model_selection의 GridSearchCV 함수이다.

# 3.1 탐색 범위 선정
# 탐색할 값들의 Argument와 범위를 정합니다.
params = {
    "n_estimators": [100, 200, 500, 1000],
    "criterion": ["mae", "mse"],
    "max_depth": [i for i in range(1, 10, 2)],
}
cv_rf_regressor = RandomForestRegressor()

# 3.2 탐색
# 탐색 시작 cv는 k-fold의 K값
grid = GridSearchCV(estimator=cv_rf_regressor, param_grid=params, cv=3, n_jobs=1)
grid = grid.fit(train_data, train_target)
print(f"Best score of paramter search is: {grid.best_score_:.4f}")
print(f"Best paramter is: {grid.best_params_}")
print("Best parameter of best score is")
for key, value in grid.best_params_.items():
    print(f"\t {key}: {value}")

# 3.3 평가
best_rf = grid.best_estimator_
cv_train_pred = best_rf.predict(train_data)
cv_test_pred = best_rf.predict(test_data)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].scatter(train_target, cv_train_pred)
axes[0].set_xlabel("predict")
axes[0].set_ylabel("real")

axes[1].scatter(test_target, cv_test_pred)
axes[1].set_xlabel("predict")
axes[1].set_ylabel("real")
plt.show()

cv_train_mse = mean_squared_error(train_target, cv_train_pred)
cv_test_mse = mean_squared_error(test_target, cv_test_pred)
print(f"Best model Train mean squared error is {cv_train_mse:.4f}")
print(f"Best model Test mean squared error is {cv_test_mse:.4f}")
cv_feature_importance = pd.Series(best_rf.feature_importances_, index=housing["feature_names"])
cv_feature_importance.sort_values(ascending=True).plot(kind="barh")
plt.show()

# 4. 마무리
print(f"Test mean squared error is {test_mse:.4f}")
print(f"Best model Test mean squared error is {cv_test_mse:.4f}")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
feature_importance.sort_values(ascending=True).plot(kind="barh", ax=axes[0])
cv_feature_importance.sort_values(ascending=True).plot(kind="barh", ax=axes[1])
plt.show()

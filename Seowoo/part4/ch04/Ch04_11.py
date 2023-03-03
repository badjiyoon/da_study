import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)

## 1. Data
# 1.1 Data Load

from sklearn.datasets import load_boston

housing = load_boston()
data, target = housing["data"], housing["target"]

# 1.2 Data EDA
pd.DataFrame(data, columns=housing["feature_names"]).describe()

print(pd.Series(target).describe())

fig, axes = plt.subplots(nrows=2, ncols=7, figsize=(20, 10))
for i, feature_name in enumerate(housing["feature_names"]):
    ax = axes[i // 7, i % 7]
    ax.scatter(data[:, i], target)
    ax.set_xlabel(feature_name)
    ax.set_ylabel("price")

# 1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

## 2. Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor()

# 2.1 Train
rf_regressor.fit(train_data, train_target)

# 2.2 Prediction
train_pred = rf_regressor.predict(train_data)
test_pred = rf_regressor.predict(test_data)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].scatter(train_target, train_pred)
axes[0].set_xlabel("predict")
axes[0].set_ylabel("real")

axes[1].scatter(test_target, test_pred)
axes[1].set_xlabel("predict")
axes[1].set_ylabel("real")

# 2.3 Validation
from sklearn.metrics import mean_squared_error

train_mse = mean_squared_error(train_target, train_pred)
test_mse = mean_squared_error(test_target, test_pred)

print(f"train mean squared error is {train_mse:.4f}")
print(f"test mean squared error is {test_mse:.4f}")

# 2.4 Feature Importance
feature_importance = pd.Series(rf_regressor.feature_importances_, index=housing["feature_names"])
feature_importance.sort_values(ascending=True).plot(kind="barh")

## 3. Best Parameter
from sklearn.model_selection import GridSearchCV

# Random Forest Regressor에서 설정하는 argument들은 다음과 같습니다.
# - n_estimators
#     - 몇 개의 의사결정나무를 생성할지 결정합니다.
# - criterion
#     - 감소 시킬 평가지표를 설정합니다.
#     - "mae": Mean Absolute Error
#     - "mse": Mean Squared Error
# - max_depth
#     - 의사결정나무가 가질 수 있는 최대 깊이를 결정합니다.

# 3.1 탐색 범위 설정
params = {
    "n_estimators": [100, 200, 500, 1000],
    "criterion": ["mae", "mse"],
    "max_depth": [i for i in range(1, 10, 2)],
}
print(params)
cv_rf_regressor = RandomForestRegressor()

# 3.2 탐색
grid = GridSearchCV(estimator=cv_rf_regressor, param_grid=params, cv=3)
grid = grid.fit(train_data, train_target)

print(f"Best score of paramter search is: {grid.best_score_:.4f}")
print(grid.best_params_)

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

cv_train_mse = mean_squared_error(train_target, cv_train_pred)
cv_test_mse = mean_squared_error(test_target, cv_test_pred)

print(f"Best model Train mean squared error is {cv_train_mse:.4f}")
print(f"Best model Test mean squared error is {cv_test_mse:.4f}")

cv_feature_importance = pd.Series(best_rf.feature_importances_, index=housing["feature_names"])
cv_feature_importance.sort_values(ascending=True).plot(kind="barh")

## 4. 마무리
print(f"Test mean squared error is {test_mse:.4f}")
print(f"Best model Test mean squared error is {cv_test_mse:.4f}")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
feature_importance.sort_values(ascending=True).plot(kind="barh", ax=axes[0])
cv_feature_importance.sort_values(ascending=True).plot(kind="barh", ax=axes[1])





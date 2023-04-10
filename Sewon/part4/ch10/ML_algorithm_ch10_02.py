#CH10_02. Boosting Regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Sample Data
data = np.linspace(0, 6, 150)[:, np.newaxis]

label = np.sin(data).ravel() + np.sin(6 * data).ravel()
noise = np.random.normal(data.shape[0]) * 0.01
label += noise

plt.scatter(data, label)
plt.show()

#1.2 Data Split
train_size = 125
train_data, test_data = data[:train_size], data[train_size:]
train_label, test_label = label[:train_size], label[train_size:]

plt.scatter(train_data, train_label)
plt.scatter(test_data, test_label, color="C1")
plt.show()

#2. Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=2)

#2.1 학습
tree.fit(train_data, train_label)

#2.2 예측
tree_train_pred = tree.predict(train_data)
tree_test_pred = tree.predict(test_data)

#2.3 평가
from sklearn.metrics import mean_squared_error

tree_train_mse = mean_squared_error(train_label, tree_train_pred)
tree_test_mse = mean_squared_error(test_label, tree_test_pred)

print(f"Tree mean squared error is {tree_train_mse:.4f}")
print(f"Tree mean squared error is {tree_test_mse:.4f}")

#2.4 시각화
plt.scatter(data, label)
plt.plot(train_data, tree_train_pred)
plt.plot(test_data, tree_test_pred)
plt.show()

#3. AdaBoost
"""
AdaBoost는 sklearn.ensemble의 AdaBoostRegressor로 생성
base_estimator 선언 필요
가장 간단한 if else로 데이터가 분류 될 수 있도록 depth가 1인 tree로 base estimator 생성
"""
from sklearn.ensemble import AdaBoostRegressor

ada_boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1))

#3.1 학습
ada_boost.fit(train_data, train_label)

#3.2 예측
ada_boost_train_pred = ada_boost.predict(train_data)
ada_boost_test_pred = ada_boost.predict(test_data)

#3.3 평가
ada_boost_train_mse = mean_squared_error(train_label, ada_boost_train_pred)
ada_boost_test_mse = mean_squared_error(test_label, ada_boost_test_pred)

print(f"Ada Boost Train mean squared error is {ada_boost_train_mse:.4f}")
print(f"Ada Boost Test mean squared error is {ada_boost_test_mse:.4f}")

#3.4 시각화
plt.scatter(data, label)
plt.plot(train_data, ada_boost_train_pred)
plt.plot(test_data, ada_boost_test_pred)
plt.show()

#4. Gradient Boost
"""
Gradient Boost는 sklearn.ensemble 의 GradientBoostingRegressor로 생성
Gradient Boost 또한 간단한 if else로 만들 수 있도록 max_depth를 1로 주겠습니다.
"""
from sklearn.ensemble import GradientBoostingRegressor

grad_boost = GradientBoostingRegressor(max_depth=1)

#4.1 학습
grad_boost.fit(train_data, train_label)

#4.2 예측
grad_boost_train_pred = grad_boost.predict(train_data)
grad_boost_test_pred = grad_boost.predict(test_data)

#4.3 평가
grad_boost_train_mse = mean_squared_error(train_label, grad_boost_train_pred)
grad_boost_test_mse = mean_squared_error(test_label, grad_boost_test_pred)

print(f"Gradient Boost Train mean squared error is {grad_boost_train_mse:.4f}")
print(f"Gradient Boost Test mean squared error is {grad_boost_test_mse:.4f}")

#4.4 시각화
plt.scatter(data, label)
plt.plot(train_data, grad_boost_train_pred)
plt.plot(test_data, grad_boost_test_pred)
plt.show()

#5. 마무리
print(f"Tree train mean squared error is {tree_train_mse:.4f}")
print(f"Ada Boost train mean squared error is {ada_boost_train_mse:.4f}")
print(f"Gradient Boost train mean squared error is {grad_boost_train_mse:.4f}")

print(f"Tree test mean squared error is {tree_test_mse:.4f}")
print(f"Ada Boost test mean squared error is {ada_boost_test_mse:.4f}")
print(f"Gradient Boost test mean squared error is {grad_boost_test_mse:.4f}")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
preds = [
    ("tree", tree_train_pred, tree_test_pred),
    ("Ada Boost", ada_boost_train_pred, ada_boost_test_pred),
    ("Gradient Boost", grad_boost_train_pred, grad_boost_test_pred)
]
for idx, (name, train_pred, test_pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(data, label)
    ax.plot(train_data, train_pred)
    ax.plot(test_data, test_pred)
    ax.set_title(name)
plt.show()
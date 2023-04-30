# 샘플 데이터와 Boostring Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
data = np.linspace(0, 6, 150)[:, np.newaxis]
label = np.sin(data).ravel() + np.sin(6 * data).ravel()
noise = np.random.normal(data.shape[0]) * 0.01
label += noise

plt.scatter(data, label)
plt.show()

# 1.2 Data Split
train_size = 125
train_data, test_data = data[:train_size], data[train_size:]
train_label, test_label = label[:train_size], label[train_size:]

plt.scatter(train_data, train_label)
plt.scatter(test_data, test_label, color="C1")
plt.show()

# 2. Decision Tree
from sklearn.tree import DecisionTreeRegressor

tree = DecisionTreeRegressor(max_depth=2)

tree.fit(train_data, train_label)

tree_train_pred = tree.predict(train_data)
tree_test_pred = tree.predict(test_data)

# 3. 평가
from sklearn.metrics import mean_squared_error

tree_train_mse = mean_squared_error(train_label, tree_train_pred)
tree_test_mse = mean_squared_error(test_label, tree_test_pred)

print(f"Tree Train mean squared error is {tree_train_mse: .4f}")
print(f"Tree Test mean squared error is {tree_test_mse: .4f}")

# 시각화
plt.scatter(data, label)
plt.plot(train_data, tree_train_pred)
plt.plot(test_data, tree_test_pred)
plt.show()

# 3. AdaBoost
from sklearn.ensemble import AdaBoostRegressor

ada_boost = AdaBoostRegressor(DecisionTreeRegressor(max_depth=1))
# 3.1 학습
ada_boost.fit(train_data, train_label)

# 3.2 예측
ada_boost_train_pred = ada_boost.predict(train_data)
ada_boost_test_pred = ada_boost.predict(test_data)

ada_boost_train_mse = mean_squared_error(train_label, ada_boost_train_pred)
ada_boost_test_mse = mean_squared_error(test_label, ada_boost_test_pred)

print(f"Ada Boost Train mean squared error is {ada_boost_train_mse: .4f}")
print(f"Ada Boost Test mean squared error is {ada_boost_test_mse: .4f}")

plt.scatter(data, label)
plt.plot(train_data, ada_boost_train_pred)
plt.plot(test_data, ada_boost_test_pred)
plt.show()

# 4. GradienBoost

from sklearn.ensemble import GradientBoostingRegressor

grad_boost = GradientBoostingRegressor(max_depth=1)

grad_boost.fit(train_data, train_label)

grad_boost_train_pred = grad_boost.predict(train_data)
grad_boost_test_pred = grad_boost.predict(test_data)

grad_boost_train_mse = mean_squared_error(train_label, grad_boost_train_pred)
gard_boost_test_mse = mean_squared_error(test_label, grad_boost_test_pred)

print(f"grad Boost Train mean squared error is {grad_boost_train_mse: .4f}")
print(f"grad Boost Test mean squared error is {gard_boost_test_mse: .4f}")

plt.scatter(data, label)
plt.plot(train_data, grad_boost_train_pred)
plt.plot(test_data, grad_boost_test_pred)
plt.show()

# 마무리
print(f"Tree Train mean squared error is {tree_train_mse: .4f}")
print(f"Ada Boost Train mean squared error is {ada_boost_train_mse: .4f}")
print(f"grad Boost Train mean squared error is {grad_boost_train_mse: .4f}")

print(f"Tree Test mean squared error is {tree_test_mse: .4f}")
print(f"Ada Boost Test mean squared error is {ada_boost_test_mse: .4f}")
print(f"grad Boost Test mean squared error is {gard_boost_test_mse: .4f}")

git, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

preds = [
    ("tree", tree_train_pred, tree_test_pred),
    ("Ada Boost", ada_boost_train_pred, ada_boost_test_pred),
    ("Gradient Boost", grad_boost_train_pred, grad_boost_test_pred),
]

for idx, (name, train_pred, test_pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(data, label)
    ax.plot(train_data, train_pred)
    ax.plot(test_data, test_pred)
    ax.set_title(name)
plt.show()

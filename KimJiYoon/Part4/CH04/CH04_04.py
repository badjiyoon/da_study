import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(2021)

# 1. Data
# 1.1 Data Load
data = np.sort(np.random.uniform(low=0, high=5, size=(80, 1)))
label = np.sin(data).ravel()
label[::5] += 3 * (0.5 - np.random.uniform(0, 1, 16))

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor='black', c='darkorange')
plt.show()

# 1.2 Viz Data
# 시각화 데이터 생성
viz_test_data = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
print('vis_test_data', viz_test_data[:5])

# Decion Tree Regressor
# Tree 분할에 따라 예측
# 2.1 분할이 없을 경우
# 평균으로 예측
viz_test_pred = np.repeat(label.mean(), len(viz_test_data))
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolors='black', c='darkorange')
plt.plot(viz_test_data, viz_test_pred, color='C2')
plt.show()

# 분할이 없을 떄 MSE variance를 계산
train_pred = np.repeat(label.mean(), len(data))
mse_var = np.var(label - train_pred)

print(f'no divide mse variance:{mse_var:.3f}')
# 첫 번쨰 분할
first_divide = DecisionTreeRegressor(max_depth=1)
first_divide.fit(data, label)

first_divide_pred = first_divide.predict(viz_test_data)

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")
plt.plot(viz_test_data, first_divide_pred, color='C2')
plt.show()

plot_tree(first_divide)
plt.show()

second_divide = DecisionTreeRegressor(max_depth=2)
second_divide.fit(data, label)
second_divide_pred = second_divide.predict(viz_test_data)
print(second_divide.tree_.threshold)

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolors='black', c='darkorange')
plt.plot(viz_test_data, second_divide_pred, color='C2')
plt.show()

plot_tree(second_divide)
plt.show()

# 3. Depth에 따른 변화
shallow_depth_tree = DecisionTreeRegressor(max_depth=2)
deep_depth_tree = DecisionTreeRegressor(max_depth=5)

shallow_depth_tree.fit(data, label)
deep_depth_tree.fit(data, label)

shallow_pred = shallow_depth_tree.predict(viz_test_data)
deep_pred = deep_depth_tree.predict(viz_test_data)

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolors='black', c='darkorange')
plt.plot(viz_test_data, shallow_pred, color='C2', label='shallow')
plt.plot(viz_test_data, deep_pred, color='C3', label='deep')
plt.show()


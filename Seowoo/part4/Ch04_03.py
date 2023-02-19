# 샘플 데이터와 Decision Tree Regressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)


## 1. Data
### 1.1 Data Load
# 예시에서 사용할 샘플 데이터를 생성합니다.
data = np.sort(np.random.uniform(low=0, high=5, size=(80, 1)))
label = np.sin(data).ravel()
label[::5] += 3 * (0.5 - np.random.uniform(0, 1, 16))

#

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.show()

### 1.2 Viz Data
# 시각화를 위한 데이터도 생성합니다.
# 0부터 5사이의 데이터를 0.01 간격으로 생성
viz_test_data = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
viz_test_data[:5]

## 2. Decion Tree Regressor
# Tree의 분할이 이루어질 때마다 어떻게 예측을 하는지 알아보겠습니다.
from sklearn.tree import DecisionTreeRegressor, plot_tree

### 2.1 분할이 없을 경우
# 분할이 없는 경우에는 학습 데이터의 평균으로 예측을 합니다.

viz_test_pred = np.repeat(label.mean(), len(viz_test_data))

# Plot으로 보면 초록색 선이 예측 한 값

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, viz_test_pred, color="C2")
plt.show()

# 분할이 없을때의 mse variance를 계산하면 다음과 같습니다.

train_pred = np.repeat(label.mean(), len(data))
train_pred
mse_var = np.var(label - train_pred)
mse_var


print(f"no divide mse variance: {mse_var:.3f}")

### 2.2 첫 번째 분할

first_divide = DecisionTreeRegressor(max_depth=1)
first_divide.fit(data, label)
first_divide_pred = first_divide.predict(viz_test_data)

# 분할하는 기준
first_divide.tree_.threshold

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")
plt.show()
# 분할이 이루어진 각 영역에서 다시 평균을 계산합니다.

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")
plt.plot(viz_test_data, first_divide_pred, color="C2")
plt.show()
# Treef를 시각화하기 위해서는 `plot_tree` 함수를 이용하면 됩니다.
# Tree를 시각화하면 아래와 같습니다.

plot_tree(first_divide)

### 2.3 두 번째 분할

second_divide = DecisionTreeRegressor(max_depth=2)
second_divide.fit(data, label)
second_divide_pred = second_divide.predict(viz_test_data)

second_divide.tree_.threshold

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, second_divide_pred, color="C2")
plot_tree(second_divide)
plt.show()
## 3. Depth에 따른 변화

shallow_depth_tree = DecisionTreeRegressor(max_depth=2)
deep_depth_tree = DecisionTreeRegressor(max_depth=5)

shallow_depth_tree.fit(data, label)
deep_depth_tree.fit(data, label)

shallow_pred = shallow_depth_tree.predict(viz_test_data)
deep_pred = deep_depth_tree.predict(viz_test_data)

# depth 5 는 빨간색 선인데 이상 데이터를 맞추기 위해 모양이 불규칙
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, shallow_pred, color="C2", label="shallow")
plt.plot(viz_test_data, deep_pred, color="C3", label="deep")
plt.legend()
plt.show()
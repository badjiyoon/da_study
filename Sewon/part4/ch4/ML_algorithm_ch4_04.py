#CH04_04. Decision Tree Regression 실습 (Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load 
 #1) 예시에서 사용할 샘플 데이터를 생성합니다.
data = np.sort(np.random.uniform(low=0, high=5, size=(80, 1)))
label = np.sin(data).ravel()
label[::5] += 3 * (0.5 - np.random.uniform(0, 1, 16))

 #2) 데이터는 하나의 변수를 가지며 변수에 따른 정답은 아래처럼 생겼습니다.
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.show()

#1.2 Viz Data
viz_test_data = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
viz_test_data[:5]

#2. Decion Tree Regressor
#Tree의 분할이 이루어질 때마다 어떻게 예측을 하는지 알아보겠습니다.

from sklearn.tree import DecisionTreeRegressor, plot_tree

#2.1 분할이 없을 경우
#분할이 없는 경우에는 학습 데이터의 평균으로 예측을 합니다.
viz_test_pred = np.repeat(label.mean(), len(viz_test_data)) #분할 전이므로 전체 평균

#Plot으로 보면 강의에서 본 하나의 선이 생깁니다.
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, viz_test_pred, color="C2")
plt.show()

#분할이 없을때의 mse variance를 계산하면 다음과 같습니다.
train_pred = np.repeat(label.mean(), len(data))
mse_var = np.var(label - train_pred)
train_pred
mse_var

print(f"no divide mse variance: {mse_var:.3f}")

#2.2 첫 번째 분할
first_divide = DecisionTreeRegressor(max_depth=1)
first_divide.fit(data, label)
first_divide 
type(first_divide)

#분할 기준
first_divide.tree_.threshold

first_divide_pred = first_divide.predict(viz_test_data)
first_divide_pred
type(first_divide_pred)

#첫번째로 분할되서 나누어진 영역을 그리면 아래와 같습니다.
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")

#분할이 이루어진 각 영역에서 다시 평균을 계산합니다.
plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.axvline(first_divide.tree_.threshold[0], color="red")
plt.plot(viz_test_data, first_divide_pred, color="C2")
plt.show()

#Tree를 시각화하기 위해서는 plot_tree 함수를 이용하면 됩니다.
plot_tree(first_divide)

#2.3 두 번째 분할
second_divide = DecisionTreeRegressor(max_depth=2)

second_divide.fit(data, label)

#분할 기준
second_divide.tree_.threshold

second_divide_pred = second_divide.predict(viz_test_data)

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, second_divide_pred, color="C2")
plt.show()

plot_tree(second_divide)

#3. Depth에 따른 변화
shallow_depth_tree = DecisionTreeRegressor(max_depth=2)
deep_depth_tree = DecisionTreeRegressor(max_depth=5)

shallow_depth_tree.fit(data, label)
deep_depth_tree.fit(data, label)

shallow_pred = shallow_depth_tree.predict(viz_test_data)
deep_pred = deep_depth_tree.predict(viz_test_data)

plt.figure(figsize=(8, 8))
plt.scatter(data, label, edgecolor="black", c="darkorange")
plt.plot(viz_test_data, shallow_pred, color="C2", label="shallow")
plt.plot(viz_test_data, deep_pred, color="C3", label="deep")
plt.legend() #범례 표시
plt.show()

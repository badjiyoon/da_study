# 샘플 데이터와 Decision Tree Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)
## 1. Data
### 1.1 Sample Data
data = {
    "value": [30, 120, 150, 390, 400, 300, 500],
    "label": [0, 0, 1, 0, 0, 1, 0]
}
data = pd.DataFrame(data)
data
## 2. Decision Tree 구현
### 2.1 변수 값에 따라 데이터를 정렬한다.
sorted_data = data.sort_values(by="value")
sorted_data
sorted_data = sorted_data.reset_index(drop=True)
sorted_data
### 2.2 정답이 바뀌는 경계 지점을 찾는다.
boundary = sorted_data["label"].diff() != 0
print(boundary)
boundary[0] = False
print(boundary)
boundary_idx = boundary.loc[boundary].index
print(boundary_idx)
### 2.3 경계의 평균값을 기준으로 잡는다.
# 첫 번째 경계 구간
idx_1 = boundary_idx[0]
data.loc[[idx_1-1, idx_1]]
bound_value_1 = data.loc[[idx_1-1, idx_1], "value"].mean()
bound_value_1
# 두 번째 경계 구간
idx_2 = boundary_idx[1]
bound_value_2 = data.loc[[idx_2-1, idx_2], "value"].mean()
bound_value_2
### 2.4 구간별 경계값을 기준으로 정보 이득을 계산한다.

def gini_index(label):
    p1 = (label == 0).mean()
    p2 = 1 - p1
    return 1 - (p1 ** 2 + p2 **2)

def concat_gini_index(left, right):
    left_gini = gini_index(left)
    right_gini = gini_index(right)
    all_num = len(left) + len(right)
    left_gini *= len(left) / all_num
    right_gini *= len(right) / all_num
    return left_gini + right_gini

# 135를 경계로 나눌 때
left_1 = sorted_data.loc[:idx_1 - 1, "label"]
right_1 = sorted_data.loc[idx_1:, "label"]
print(left_1)
gini_index(right_1)
print(concat_gini_index(left_1, right_1))

# 345를 경계로 나눌 때
left_2 = sorted_data.loc[:idx_2 - 1, "label"]
right_2 = sorted_data.loc[idx_2:, "label"]
print(concat_gini_index(left_2, right_2))

## 3. Decision Tree Package
from sklearn.tree import DecisionTreeClassifier, plot_tree

tree = DecisionTreeClassifier(max_depth=1)
tree.fit(data["value"].to_frame(), data["label"])
plt.figure(0)
plot_tree(tree)
plt.show()


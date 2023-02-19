import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
data = {
    'value': [30, 120, 150, 390, 400, 300, 500],
    'label': [0, 0, 1, 0, 0, 1, 0]
}

data = pd.DataFrame(data)
# 2.Decision Tree 구현
# 2.2 변수 값에 따라 데이터를 정렬한다.
print(data.sort_values(by='label'))
sorted_data = data.sort_values(by='value')
print('sorted_data : ', sorted_data)

# 인덱스 값이 안맞기 때문에 인덱스 정렬 후 초기화
sorted_data = sorted_data.reset_index(drop=True)

# 2.2 정답이 바뀌는 경계 지점을 찾는다.
# 바로 전 데이터에서 값을 뺸다.
print('diff : ', sorted_data['label'].diff())
boundary = sorted_data['label'].diff() != 0
print('boundary label : ', boundary)
# 처음값은 NAN기 떄문에 경계값으로 보지 않는다.
boundary[0] = False
print('boundary [0] : ', boundary)
boundary_idx = boundary.loc[boundary].index
print('boundary_idx : ', boundary_idx)
# 2.3 경계의 평균값을 기준으로 잡는다.
# 첫 번째 경계 구간
idx_1 = boundary_idx[0]
print('[idx_1 - 1, idx_1] : ', [idx_1 - 1, idx_1])
bound_value_1 = data.loc[[idx_1 - 1, idx_1], 'value'].mean()
print('bound_value_1 : ', bound_value_1)
# 두 번쨰 경계 구간
idx_2 = boundary_idx[1]
bound_value_2 = data.loc[[idx_2 - 1, idx_2], 'value'].mean()


# 2.4 구간별 경계값을 기준으로 정보 이득을 계산한다.
def gini_index(label):
    # 레이블이 0인 갯수
    p1 = (label == 0).mean()
    p2 = 1 - p1
    return 1 - (p1 ** 2 + p2 ** 2)


def concat_gini_index(left, right):
    left_gini = gini_index(left)
    right_gini = gini_index(right)
    all_num = len(left) * len(right)
    left_gini *= len(left) / all_num
    right_gini *= len(right) / all_num
    return left_gini + right_gini


# 135를 경계로 나눌 떄
left_1 = sorted_data.loc[:idx_1 - 1, 'label']
right_1 = sorted_data.loc[:idx_1 - 1, 'label']

print('left_1 : ', left_1)
print('right_1 : ', right_1)
print('concat_gini_index : ', concat_gini_index(left_1, right_1))

# 345를 경계로 나눌 때
left_2 = sorted_data.loc[:idx_2 - 1, 'label']
right_2 = sorted_data.loc[:idx_2 - 1, 'label']
print('concat_gini_index : ', concat_gini_index(left_2, right_2))

# 3. Decision Tree Package
tree = DecisionTreeClassifier(max_depth=1)
tree.fit(data['value'].to_frame(), data['label'])
plot_tree(tree)
plt.show()

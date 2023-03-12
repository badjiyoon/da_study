import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

np.random.seed(2021)

# 1 Data
# 1.1 Dta Load
iris = load_iris()

data = iris.data
target = iris.target
# 데이터의 일부만 사용함
data = data[target != 0, 2:]
target = target[target != 0]

data = pd.DataFrame(data)
target = pd.DataFrame(target)

# 데이터 중복 제거
target = target.loc[~data.duplicated()].values.flatten()
data = data.loc[~data.duplicated()].values

# 스캐터 차트 출력
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()

# 1.2 시각화 데이터
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 2. k 값에 따른 결정 경계
from sklearn.neighbors import KNeighborsClassifier

# k 값에 따른 knn의 결정경계를 그려봅니다.
# k 가 작을수록 overfitting이 k가 클수록 underfitting이 됩니다.
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for idx, n in enumerate(range(1, 12, 2)):
    # knn 생성 및 학습
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(data, target)

    # 시각회 데이터 예측
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax = axes[idx // 3, idx % 3]

    # 영역 표시
    ax.contourf(xx, yy, Z)

    # 데이터 표시
    ax.scatter(
        data[:, 0], data[:, 1], c=target, alpha=1.0, edgecolor="black"
    )
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel(iris.feature_names[0])
    ax.set_ylabel(iris.feature_names[1])
    ax.set_title(f"{n} Nearest Neighbors")

plt.show()

# 3. 나의 가장 가까운 이웃은?
# KNN의 거리의 종류는 p를 통해서 바꿀 수 있습니다.
# - p=1
#     - 맨해튼 거리
# - p=2
#     - 유클리드 거리

# 3.1 Euclidean Distance
train_data, train_target = data[:-1], target[:-1]
test_data = data[-1:]

len(train_data), len(test_data)

# 거리를 두기위해 둔다
euclid_knn = KNeighborsClassifier(n_neighbors=10)
euclid_knn.fit(train_data, train_target)

# 가장 가까운값을 리턴해줌
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=10, p=2,
                     weights='uniform')
# 제일 가까운 것 하나만 리턴
euclid_knn.kneighbors(
    test_data, n_neighbors=1, return_distance=False
).ravel()
# 불필요한 Shape 제거 -> ravel
euclid_neighbors_idx = euclid_knn.kneighbors(
    test_data, n_neighbors=10, return_distance=False
).ravel()
euclid_neighbors = train_data[euclid_neighbors_idx]
euclid_neighbors_label = train_target[euclid_neighbors_idx]
test_data
euclid_neighbors
euclid_neighbors_label
euclid_knn.predict(test_data)
euclid_knn.predict_proba(test_data)

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(euclid_neighbors[:, 0], euclid_neighbors[:, 1], c=euclid_neighbors_label, edgecolors="red", s=500)
plt.show()

# 3.2 Manhattan Distance
manhattan_knn = KNeighborsClassifier(n_neighbors=10, p=1)
manhattan_knn.fit(train_data, train_target)

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=10, p=1,
                     weights='uniform')

manhattan_neighbors_idx = manhattan_knn.kneighbors(
    test_data, n_neighbors=10, return_distance=False
).ravel()
manhattan_neighbors = train_data[manhattan_neighbors_idx]
manhattan_neighbors_label = train_target[manhattan_neighbors_idx]
manhattan_neighbors
manhattan_neighbors_label
manhattan_knn.predict_proba(test_data)

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(manhattan_neighbors[:, 0], manhattan_neighbors[:, 1], c=manhattan_neighbors_label, edgecolors="red", s=500)
plt.show()

# 3.3 비교
euclid_neighbors_idx
manhattan_neighbors_idx
# 중복값 제거 -> 양쪽으로 셋으로 처리
set(euclid_neighbors_idx) - set(manhattan_neighbors_idx)
diff_neighbors_idx = list(set(euclid_neighbors_idx) - set(manhattan_neighbors_idx))
diff_neighbors_idx.extend(list(set(manhattan_neighbors_idx) - set(euclid_neighbors_idx)))
diff_neighbors_idx

diff_neighbors = train_data[diff_neighbors_idx]
diff_neighbors_label = train_target[diff_neighbors_idx]

same_neighbors_idx = list(set(euclid_neighbors_idx) & set(manhattan_neighbors_idx))
same_neighbors_idx

same_neighbors = train_data[same_neighbors_idx]
same_neighbors_label = train_target[same_neighbors_idx]

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(diff_neighbors[:, 0], diff_neighbors[:, 1], c=diff_neighbors_label, edgecolors="red", s=500)
plt.scatter(same_neighbors[:, 0], same_neighbors[:, 1], c=same_neighbors_label, edgecolors="blue", s=500)
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
target = iris.target
print(data)
print(target)

print(data.shape)
# 두 가지로만 분류하기 위히 0이 아닌 값들만 사용
data = data[target != 0, 2:]
target = target[target != 0]

data = pd.DataFrame(data)
target = pd.DataFrame(target)

print(~data.duplicated())
# 중복 데이터 제거
target = target.loc[~data.duplicated()].values.flatten()
data = data.loc[~data.duplicated()].values

print(data.shape)
target
plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()

# 영역의 시각화
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
x_min
x_max
y_min
y_max



# K 값에 따른 결정 경계
from sklearn.neighbors import KNeighborsClassifier

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

# 유클리드 거리
## p = 1 : 맨해튼 거리
## p = 2 : 유클리드 거리
print(len(target[:-1]))

train_data, train_target = data[:-1], target[:-1]
test_data = data[-1:]
print(len(train_data), len(test_data))

euclid_knn = KNeighborsClassifier(n_neighbors=10)   # 홀수
euclid_knn.fit(train_data, train_target)

# 세로운 데이터의 가까운 값들을 배열로 반환
# .ravel() : 불필요한 shape 제거
euclid_knn.kneighbors(
    test_data, n_neighbors=1, return_distance=False
)

euclid_neighbors_idx = euclid_knn.kneighbors(
    test_data, n_neighbors=10, return_distance=False
).ravel()

euclid_neighbors = train_data[euclid_neighbors_idx]
euclid_neighbors_label = train_target[euclid_neighbors_idx]

print(euclid_neighbors)
print(euclid_neighbors_label)

print(euclid_knn.predict(test_data))
print(euclid_knn.predict_proba(test_data)) # 클래스에 속할 확률

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(euclid_neighbors[:, 0], euclid_neighbors[:, 1], c=euclid_neighbors_label, edgecolors="red", s=500)
plt.show()

# 맨해튼 거리
manhattan_knn = KNeighborsClassifier(n_neighbors=10, p=1)
manhattan_knn.fit(train_data, train_target)

manhattan_neighbors_idx = manhattan_knn.kneighbors(
    test_data, n_neighbors=10, return_distance=False
).ravel()
manhattan_neighbors = train_data[manhattan_neighbors_idx]
manhattan_neighbors_label = train_target[manhattan_neighbors_idx]

print(manhattan_neighbors)
print(manhattan_neighbors_label)
print(manhattan_knn.predict_proba(test_data))

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(manhattan_neighbors[:, 0], manhattan_neighbors[:, 1], c=manhattan_neighbors_label, edgecolors="red", s=500)
plt.show()

# 비교
print(euclid_neighbors_idx)
print(manhattan_neighbors_idx)
print(set(euclid_neighbors_idx) - set(manhattan_neighbors_idx))
print(set(manhattan_neighbors_idx) - set(euclid_neighbors_idx))
# 유니크 값을 set으로 준다
diff_neighbors_idx = list(set(euclid_neighbors_idx) - set(manhattan_neighbors_idx))
diff_neighbors_idx.extend(list(set(manhattan_neighbors_idx) - set(euclid_neighbors_idx)))
print(diff_neighbors_idx)

diff_neighbors = train_data[diff_neighbors_idx]
diff_neighbors_label = train_target[diff_neighbors_idx]

same_neighbors_idx = list(set(euclid_neighbors_idx) & set(manhattan_neighbors_idx))
print(same_neighbors_idx) # 같았던 값들

same_neighbors = train_data[same_neighbors_idx]
same_neighbors_label = train_target[same_neighbors_idx]

plt.figure(figsize=(15, 8))
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, s=500)
plt.scatter(test_data[0, 0], test_data[0, 1], marker="*", s=1000)
plt.scatter(diff_neighbors[:, 0], diff_neighbors[:, 1], c=diff_neighbors_label, edgecolors="red", s=500)
plt.scatter(same_neighbors[:, 0], same_neighbors[:, 1], c=same_neighbors_label, edgecolors="blue", s=500)
plt.show()

# 파란색이 유클리드와 맨해튼에서 같은 값들
# 빨간색이 거리가 다르게 나왔던 값들
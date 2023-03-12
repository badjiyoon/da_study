#CH06_02. KNN 실습(Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
target = iris.target

target
target != 0 #target에서 0이 아닌 값 찾기 (Boolean)

data.shape #(150, 4)

data=data[target !=0, 2:] #data에서도 3, 4번째 변수만 사용(indexing)
target=target[target != 0] #target도 0 아닌 값만 사용

data.shape #(100, 2)

data = pd.DataFrame(data) 
target = pd.DataFrame(target)
"""
1, 2로 바뀐 데이터들을 DataFrame으로 변환
array인 상태에서는 duplicated 적용 불가 -> DataFrame으로 변환
"""

data.loc[data.duplicated()]
"""
1, 2번 변수는 삭제했기 때문에 중복값 발생
중복값 삭제: duplicated()
~data.duplicated(): 중복이 아닌 값만 둔다.
"""
target = target.loc[~data.duplicated()].values.flatten()
data = data.loc[~data.duplicated()].values
#values 속성의 다차원 배열을 flatten 메소드로 1차원으로 변형

data.shape #(80, 2)

plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()

#1.2 시각화 데이터
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
plt.show()

#2. k 값에 따른 결정 경계

from sklearn.neighbors import KNeighborsClassifier
"""
k값에 따른 knn의 결정경계 그리기
k가 작을수록 overfitting
k가 클수록 underfitting
"""

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for idx, n in enumerate(range(1, 12, 2)):
    # knn 생성 및 학습
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(data, target)

    # 시각회 데이터 예측
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax = axes[idx//3, idx%3]

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

#3. 가장 가까운 이웃(Nearest Neighbor)
"""
KNN의 거리의 종류는 p를 통해서 바꿀 수 있습니다.
p=1 : 맨해튼 거리
p=2 : 유클리드 거리
"""

#3.1 Euclidean Distance
train_data, train_target = data[:-1], target[:-1]
test_data = data[-1:]

len(train_data)
len(test_data)

euclid_knn = KNeighborsClassifier(n_neighbors=10)
euclid_knn.fit(train_data, train_target)

euclid_knn.kneighbors(
    test_data, n_neighbors=1, return_distance=False
).ravel()

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

#3.2 Manhattan Distance
manhattan_knn = KNeighborsClassifier(n_neighbors=10, p=1)
manhattan_knn.fit(train_data, train_target)

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

#3.3 비교
euclid_neighbors_idx
manhattan_neighbors_idx

set(euclid_neighbors_idx) - set(manhattan_neighbors_idx)
set(manhattan_neighbors_idx) - set(euclid_neighbors_idx)

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
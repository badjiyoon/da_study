import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
from sklearn.datasets import make_blobs

# https://datascienceschool.net/03%20machine%20learning/09.02%20%EB%B6%84%EB%A5%98%EC%9A%A9%20%EA%B0%80%EC%83%81%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%83%9D%EC%84%B1.html
# 가상 데이터 생성
data, label = make_blobs(n_samples=1500, random_state=170)
plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()

# 2. K Means
# 2.1 정확한 군집의 갯수를 맞춘 경우
from sklearn.cluster import KMeans

# 정확한 군집 갯수로 클러스터를 잡는다.
correct_kmeans = KMeans(n_clusters=3)
correct_kmeans.fit(data)
correct_pred = correct_kmeans.predict(data)
# K Means는 중심을 찾아놓고 새로운 데이터가 들어왔을 때 중심을 비교하고 군집을 만들어주는 알고리즘
# 각 중심이 저장되는 장소
correct_center = correct_kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=correct_pred)
# 빨간색(*)이 군집의 중심
plt.scatter(correct_center[:, 0], correct_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2 군집의 갯수를 틀린 경우
# 2.2.1 적은 경우
small_kmeans = KMeans(n_clusters=2)
small_kmeans.fit(data)
small_pred = small_kmeans.predict(data)
small_center = small_kmeans.cluster_centers_

# 적은 경우를 보게되면 중심축이 2번과 3번사이에 있으며 2,3 군집이 하나로됨
plt.scatter(data[:, 0], data[:, 1], c=small_pred)
plt.scatter(small_center[:, 0], small_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2.2 큰 경우
large_kmeans = KMeans(n_clusters=4)
large_kmeans.fit(data)
large_pred = large_kmeans.predict(data)
large_center = large_kmeans.cluster_centers_

# 2개의 군집에서는 정확하게 찾았지만 하나의 군집이 반으로 나누어지는 케이스가 됬다.
plt.scatter(data[:, 0], data[:, 1], c=large_pred)
plt.scatter(large_center[:, 0], large_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2.3 적절한 K를 찾기
# 매트릭스 값
sse_per_n = []
# 1 ~ 12중 가장 최적의 클러스터 값을 찾자
for n in range(1, 12, 2):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    sse = kmeans.inertia_
    sse_per_n += [sse]

# 3에서 그래프가 꺽임
plt.plot(range(1, 12, 2), sse_per_n)
plt.title("Sum of Squared Error")
plt.show()

# 3. K Means의 한계
# 3.1 서로 다른 크기의 군집
size_data = size_label = make_blobs(
    n_samples=1500,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=170
)

size_data = np.vstack(
    (size_data[size_label == 0][:500],
     size_data[size_label == 1][:100],
     size_data[size_label == 2][:10])
)

size_label = [0] * 500 + [1] * 100 + [2] * 10

plt.scatter(size_data[:, 0], size_data[:, 1], c=size_label)
plt.show()

# 퍼져있는 데이터 파악안됨
size_kmeans = KMeans(n_clusters=3, random_state=2021)
size_pred = size_kmeans.fit_predict(size_data)
size_center = size_kmeans.cluster_centers_

plt.scatter(size_data[:, 0], size_data[:, 1], c=size_pred)
plt.scatter(size_center[:, 0], size_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 3.2 서로 다른 밀도의 군집
denstiny_data, density_label = make_blobs(
    n_samples=1500,
    cluster_std=[1.0, 2.5, 0.5],
    random_state=170
)

plt.scatter(denstiny_data[:, 0], denstiny_data[:, 1], c=density_label)
plt.show()

density_kmeans = KMeans(n_clusters=3, random_state=2021)
density_pred = density_kmeans.fit_predict(denstiny_data)
density_center = density_kmeans.cluster_centers_

# 작은 크기에 밀도에 있는 데이터가 군집이 잘 안됨
plt.scatter(data[:, 0], data[:, 1], c=density_pred)
plt.scatter(density_center[:, 0], density_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 3.3 지역적 패턴이 있는 군집
transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
pattern_data = np.dot(data, transformation)
plt.scatter(pattern_data[:, 0], pattern_data[:, 1], c=label)
plt.show()

pattern_kmeans = KMeans(n_clusters=3, random_state=2021)
pattern_pred = pattern_kmeans.fit_predict(pattern_data)
pattern_center = pattern_kmeans.cluster_centers_

# 대각선으로 되어있는 데이터는 군집에 중심점이 이상해짐
plt.scatter(data[:, 0], data[:, 1], c=pattern_pred)
plt.scatter(pattern_center[:, 0], pattern_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 4. DBSCAN -> 밀도 기반으로 KMeans의 단점을 처리
from sklearn.cluster import DBSCAN

# 4.1 서로 다른 크기의 군집 -> DBSCAN의 경우 노이즈로 필터링 했음
size_dbscan = DBSCAN(eps=1.0)
size_db_pred = size_dbscan.fit_predict(size_data)
plt.scatter(size_data[:, 0], size_data[:, 1], c=size_db_pred)
plt.show()

# 4.2 서로 다른 밀도의 군집 -> 큰 군집들으 분류했음
density_dbscan = DBSCAN()
density_db_pred = density_dbscan.fit_predict(denstiny_data)
plt.scatter(denstiny_data[:, 0], denstiny_data[:, 1], c=density_db_pred)
plt.show()

# 4.3 지역적 패턴이 있는 군집 -> 예상대로 군집 되고 노이즈로 필터링
pattern_db = DBSCAN(eps=.3, min_samples=20)
pattern_db_pred = pattern_db.fit_predict(pattern_data)
plt.scatter(pattern_data[:, 0], pattern_data[:, 1], c=pattern_db_pred)
plt.show()

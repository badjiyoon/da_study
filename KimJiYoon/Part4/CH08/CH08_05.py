import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
from sklearn.datasets import make_blobs

data, label = make_blobs(n_samples=1500, random_state=170)
plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()

# 2. K Means
# 2.1 정확한 군집의 갯수를 맞춘 경우
from sklearn.cluster import KMeans

correct_kmeans = KMeans(n_clusters=3)
correct_kmeans.fit(data)
correct_pred = correct_kmeans.predict(data)
correct_center = correct_kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=correct_pred)
plt.scatter(correct_center[:, 0], correct_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2 군집의 갯수를 틀린 경우
# 2.2.1 적은 경우
small_kmeans = KMeans(n_clusters=2)
small_kmeans.fit(data)
small_pred = small_kmeans.predict(data)
small_center = small_kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=small_pred)
plt.scatter(small_center[:, 0], small_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2.2 큰 경우
large_kmeans = KMeans(n_clusters=4)
large_kmeans.fit(data)
large_pred = large_kmeans.predict(data)
large_center = large_kmeans.cluster_centers_

plt.scatter(data[:, 0], data[:, 1], c=large_pred)
plt.scatter(large_center[:, 0], large_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 2.2.3 적절한 K를 찾기
sse_per_n = []
for n in range(1, 12, 2):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(data)
    sse = kmeans.inertia_
    sse_per_n += [sse]

plt.plot(range(1, 12, 2), sse_per_n)
plt.title("Sum of Squared Error")

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

plt.scatter(data[:, 0], data[:, 1], c=pattern_pred)
plt.scatter(pattern_center[:, 0], pattern_center[:, 1], marker="*", s=100, color="red")
plt.show()

# 4. DBSCAN
from sklearn.cluster import DBSCAN

# 4.1 서로 다른 크기의 군집
size_dbscan = DBSCAN(eps=1.0)
size_db_pred = size_dbscan.fit_predict(size_data)
plt.scatter(size_data[:, 0], size_data[:, 1], c=size_db_pred)
plt.show()

# 4.2 서로 다른 밀도의 군집
density_dbscan = DBSCAN()
density_db_pred = density_dbscan.fit_predict(denstiny_data)
plt.scatter(denstiny_data[:, 0], denstiny_data[:, 1], c=density_db_pred)
plt.show()

# 4.3 지역적 패턴이 있는 군집
pattern_db = DBSCAN(eps=.3, min_samples=20)
pattern_db_pred = pattern_db.fit_predict(pattern_data)
plt.scatter(pattern_data[:, 0], pattern_data[:, 1], c=pattern_db_pred)
plt.show()

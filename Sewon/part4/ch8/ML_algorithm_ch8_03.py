#CH08_03. Hierachical Clustering 실습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Sample data
data = np.array(
    [
        (1, 5),
        (2, 4),
        (4, 6),
        (4, 3),
        (5, 3),
    ]
)
data

#2. Hierarchical Clustering
"""
Hierarchical Clustering은 sklearn.cluster의 AgglomerativeClustering를 이용

1) 최단 연결법
2) 최장 연결법
3) 평균 연결법
4) 중심 연결법
-> linkage argument

average: 평균 연결법
complete: 최장 연결법
single: 최단 연결법
ward: 중심 연결법
-> 기본 값은 ward
"""

#2.1 학습
from sklearn.cluster import AgglomerativeClustering

single_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage="single"
) 
single_cluster.fit(data) 

#2.2 Dendrogram
from scipy.cluster.hierarchy import dendrogram

def plot_dendrogram(model, **kwargs):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    dendrogram(linkage_matrix, **kwargs, labels=["A", "B", "C", "D", "E"])

plt.title('Hierarchical Clustering Dendrogram with single linkage')
plot_dendrogram(single_cluster, truncate_mode='level', p=3)
plt.show() #최단연결법: single cluster

#2.3 여러 개의 클러스터

#2.3.1 2개의 클러스터
single_cluster_2 = AgglomerativeClustering(
    n_clusters=2, linkage="single"
)

single_cluster_2.fit(data)
single_cluster_2.labels_

plt.figure(figsize=(7, 7))
plt.scatter(data[:, 0], data[:, 1], c=single_cluster_2.labels_)
for i, txt in enumerate(["A", "B", "C", "D", "E"]):
    plt.annotate(txt, (data[i, 0], data[i, 1]))
plt.show()

#2.3.2 3개의 클러스터
single_cluster_3 = AgglomerativeClustering(
    n_clusters=3, linkage="single"
)

single_cluster_3.fit(data)
single_cluster_3.labels_

plt.figure(figsize=(7, 7))
plt.scatter(data[:, 0], data[:, 1], c=single_cluster_3.labels_)
for i, txt in enumerate(["A", "B", "C", "D", "E"]):
    plt.annotate(txt, (data[i, 0], data[i, 1]))
plt.show()

#3. 다른 연결법

#3.1 평균 연결법
avg_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage="average"
)

avg_cluster.fit(data)

plt.title('Hierarchical Clustering Dendrogram with Average linkage')
plot_dendrogram(avg_cluster, truncate_mode='level', p=3)
plt.show()

#3.2 최장 연결법
max_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage="complete"
)

max_cluster.fit(data)

plt.title('Hierarchical Clustering Dendrogram with Maximum linkage')
plot_dendrogram(max_cluster, truncate_mode='level', p=3)
plt.show()

#3.3 중심 연결법
centroid_cluster = AgglomerativeClustering(
    distance_threshold=0, n_clusters=None, linkage="ward"
)

centroid_cluster.fit(data)

plt.title('Hierarchical Clustering Dendrogram with Centorid linkage')
plot_dendrogram(centroid_cluster, truncate_mode='level', p=3)
plt.show()

#4. 마무리
clusters = [
    ("Single", single_cluster),
    ("Average", avg_cluster),
    ("Maximum", max_cluster),
    ("Centroid", centroid_cluster),
]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

for idx, (name, cluster) in enumerate(clusters):
    ax = axes[idx//2, idx%2]
    ax.set_title(f'Hierarchical Clustering Dendrogram with {name} linkage')
    plot_dendrogram(cluster, truncate_mode='level', p=3, ax=ax)
plt.show()
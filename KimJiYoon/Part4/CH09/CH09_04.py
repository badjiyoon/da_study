# 차원 축소와 군집화
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1. Data Load
from sklearn.datasets import load_digits
digits = load_digits()

data, target = digits["data"], digits["target"]
data[0], target[0]

# 1.2 데이터 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
tsne_latent = tsne.fit_transform(scaled_data)

def visualize_latent_space_with_label(latent, pred):
    for label in np.unique(pred):
        index = pred == label
        component_1 = latent[index, 0]
        component_2 = latent[index, 1]
        plt.scatter(component_1, component_2, c=f"C{label}", label=label)
    plt.legend()

plt.figure(figsize=(7, 7))
visualize_latent_space_with_label(tsne_latent, target)

# 2. Clustering
# 우선 데이터에 다른 처리 없이 Kmeans로 군집화를 해보겠습니다.
# k means 의 k는 정답을 알고 있는 상태니 10을 주어서 실제로 잘 군집이 되는지 확인해 보겠습니다.
# 2.1 학습
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
kmeans.fit(scaled_data)

pred = kmeans.predict(scaled_data)
plt.figure(figsize=(7, 7))
visualize_latent_space_with_label(tsne_latent, pred)
plt.show()

# 3. PCA & Clustering
# 3.1 PCA 데이터 생성
from sklearn.decomposition import PCA

pca = PCA(n_components=12)
pca.fit(scaled_data)
pca_data = pca.transform(scaled_data)
pca_kmeans = KMeans(n_clusters=10)

pca_kmeans.fit(pca_data)

pca_pred = pca_kmeans.predict(pca_data)
plt.figure(figsize=(7, 7))
visualize_latent_space_with_label(tsne_latent, pca_pred)
plt.show()

plt.figure(figsize=(21, 7))
plt.subplot(131)
visualize_latent_space_with_label(tsne_latent, target)
plt.title("Raw TSNE")
plt.subplot(132)
visualize_latent_space_with_label(tsne_latent, pred)
plt.title("Raw Clustering")
plt.subplot(133)
visualize_latent_space_with_label(tsne_latent, pca_pred)
plt.title("PCA Clustering")
plt.show()

scaled_data.shape
pca_data.shape


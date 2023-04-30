#CH09_02. 차원 축소 시각화 실습 (Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
from sklearn.datasets import load_digits

digits = load_digits()

data, target = digits["data"], digits["target"]

data[0], target[0]
data[0].shape #(64, )

samples = data[:10].reshape(10, 8, 8)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    axes[idx//5, idx%5].imshow(sample, cmap="gray")
plt.show()

#1.2 데이터 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaled_data = scaler.fit_transform(data)

#2. PCA
#2.1 학습
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(scaled_data)

#2.2 설명된 분산
"PCA는 첫 번째 주성분이 가장 크며 갈수록 작아진다."
pca.explained_variance_
plt.plot(pca.explained_variance_)
plt.show()

#2.3 설명된 분산의 비율
pca.explained_variance_ratio_
plt.plot(pca.explained_variance_ratio_)
plt.plot(pca.explained_variance_ratio_.cumsum(), linestyle="--")
plt.show()

#3. 제한된 PCA
#3.1 비율로 사용하는 방법
"""
n_components argument는 int와 float을 입력
0~1사이가 들어올 경우 설명된 분산이 해당 값에 도달할 때까지 주성분을 선택
"""
ratio_pca = PCA(n_components=0.8)
ratio_pca.fit(scaled_data)

ratio_pca.explained_variance_ratio_
ratio_pca.explained_variance_ratio_.cumsum() #선택된 주성분 21개

ratio_pca.n_components_

#3.2 개수를 지정해서 사용하는 방법
"""
n_components argument는 int와 float을 입력
int 값으로 1보다 큰 값을 줄 경우에는 지정된 개수만큼의 주성분을 계산
"""
plt.plot(pca.explained_variance_)

n_comp_pca = PCA(n_components=8) #elbow point인 8개만 주성분으로 선택
n_comp_pca.fit(scaled_data)

n_comp_pca.explained_variance_ratio_
n_comp_pca.explained_variance_ratio_.cumsum()
n_comp_pca.n_components_

#3.3 시각화
"""
사람이 인식하는 최대 차원의 크기는 3차원이므로 2차원 또는 3차원으로 데이터 축소하여 시각화
2차원으로 차원 축소는 주성분의 개수를 2개로, 3차원으로 차원 축소는 주성분의 개수를 3개로 함
"""
viz_pca = PCA(n_components=2)
viz_pca_latent = viz_pca.fit_transform(scaled_data)

def visualize_latent_space_with_label(latent):
    for label in np.unique(target):
        index = target == label
        component_1 = latent[index, 0]
        component_2 = latent[index, 1]
        plt.scatter(component_1, component_2, c=f"C{label}", label=label)
    plt.legend()
plt.show()

visualize_latent_space_with_label(viz_pca_latent)

#4. LDA
"PCA와 비슷하지만 학습할 때 label"
#4.1 학습
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()

lda.fit(scaled_data, target)

#4.2 설명된 분산
"""
LDA는 eigenvalue와 같이 분산의 크기를 나타내는 값은 없음
설명된 분산의 크기만 확인 가능
"""
lda.explained_variance_ratio_

plt.plot(lda.explained_variance_ratio_)
plt.plot(lda.explained_variance_ratio_.cumsum(), linestyle="--")
plt.show()

#4.2 시각화
viz_lda = LinearDiscriminantAnalysis(n_components=2)
viz_lda_latent = viz_lda.fit_transform(scaled_data, target)

visualize_latent_space_with_label(viz_lda_latent)


#4. T-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)

#4.1 시각화
tsne_latent = tsne.fit_transform(scaled_data)
visualize_latent_space_with_label(tsne_latent)

#5.마무리
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
latents = [
    ("pca", viz_pca_latent),
    ("lda", viz_lda_latent),
    ("tsne", tsne_latent)
]
for idx, (name, latent) in enumerate(latents):
    ax = axes[idx]
    ax.scatter(latent[:, 0], latent[:, 1], c=target)
    ax.set_title(name)
plt.show()
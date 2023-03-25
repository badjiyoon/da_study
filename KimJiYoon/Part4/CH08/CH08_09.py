import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# K Means를 이용한 이미지 압축
# 1. Data
# 1.1 Data Load
from sklearn.datasets import load_sample_image
china = load_sample_image("china.jpg")

plt.imshow(china)
plt.show()

# 1.2 Data Sacling
china_flatten = china / 255.0
china_flatten = china_flatten.reshape(-1, 3)
china_flatten.shape

# 1.3 Data EDA
def plot_pixels(data, colors=None, N=10000):
    if colors is None:
        colors = data

    rng = np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors = colors[i]
    R, G, B = data[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

plot_pixels(china_flatten)
plt.show()

# 2. K Means
# 2.1 학습
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=16)
kmeans.fit(china_flatten)

# 2.2 새로운 색상
kmeans.cluster_centers_

# 2.3 변환
new_color_label = kmeans.predict(china_flatten)
new_colors = kmeans.cluster_centers_[new_color_label]
plot_pixels(china_flatten, colors=new_colors)
plt.show()

china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
# fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title("Original Image", size=16)
ax[1].imshow(china_recolored)
ax[1].set_title("16-color Image", size=16)
plt.show()

# 3. 더작은 K
# 3.1 학습
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8)
kmeans.fit(china_flatten)

# 3.2 새로운 색상
kmeans.cluster_centers_
# 3.3 변환
new_color_label = kmeans.predict(china_flatten)
new_colors = kmeans.cluster_centers_[new_color_label]
plot_pixels(china_flatten, colors=new_colors)
plt.show()

china_recolored = new_colors.reshape(china.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
# fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title("Original Image", size=16)
ax[1].imshow(china_recolored)
ax[1].set_title("8-color Image", size=16);

plt.show()

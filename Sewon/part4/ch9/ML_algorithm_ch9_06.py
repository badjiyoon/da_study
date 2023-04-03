#CH09_02. Eigenface와 얼굴 사진 분류 (Python)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data, target = faces["data"], faces["target"]

#1.2 Data EDA
n_samples, h, w = faces.images.shape
n_samples, h, w #(개수, height, width)=(1288, 50, 37)

target_names = faces.target_names
n_classes = target_names.shape[0]
target_names #이름 확인

samples = data[:10].reshape(10, h, w)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    ax = axes[idx//5, idx%5]
    ax.imshow(sample, cmap="gray")
    ax.set_title(target_names[target[idx]])
plt.show() #이미지 확인

#1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_target)}, {len(train_target)/len(data):.2f}")
print(f"test_data size: {len(test_target)}, {len(test_target)/len(data):.2f}")

#1.4 Data Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

#2. Eigenface
"""
Eigenface
-PCA를 이용해 얼굴 사진을 축소하면 생기는 eigenvector가 얼굴 모양과 같다고 하여서 생긴 용어
"""
#2.1 학습
from sklearn.decomposition import PCA

pca = PCA()

pca.fit(scaled_train_data)

plt.plot(pca.explained_variance_ratio_.cumsum())
plt.axhline(0.9, color="red", linestyle="--")
plt.show()

pca = PCA(n_components=0.9) #explained variance ratio가 0.9가 되는 지점의 component를 사용
pca.fit(scaled_train_data)

pca_train_data = pca.transform(scaled_train_data)
pca_test_data = pca.transform(scaled_test_data)

#2.2 시각화
"""
PCA로 학습한 eigen vector를 시각화
≒ PCA를 통해 다음 eigen vector에 나오는 얼굴의 특징을 추출
"""
eigenfaces = pca.components_.reshape((pca.n_components_, h, w))
samples = eigenfaces[:10].reshape(10, h, w)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    ax = axes[idx//5, idx%5]
    ax.imshow(sample, cmap="gray")
    ax.set_title(f"eigenface {idx}")
plt.show()

#3. SVM
#3.1 Raw Data
from sklearn.svm import SVC

svm = SVC()
svm.fit(scaled_train_data, train_target)

train_pred = svm.predict(scaled_train_data)
test_pred = svm.predict(scaled_test_data)

from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

print(f"train accuracy is {train_acc:.4f}")
print(f"test accuracy is {test_acc:.4f}")

#3.2 Eigenface
"Eigenface로 추출된 특징만으로 SVM을 학습시킨 후 결과 보기"
eigenface_svm = SVC()
eigenface_svm.fit(pca_train_data, train_target)

pca_train_pred = eigenface_svm.predict(pca_train_data)
pca_test_pred = eigenface_svm.predict(pca_test_data)

pca_train_acc = accuracy_score(train_target, pca_train_pred)
pca_test_acc = accuracy_score(test_target, pca_test_pred)

print(f"Eigenface train accuracy is {pca_train_acc:.4f}")
print(f"Eigenface test accuracy is {pca_test_acc:.4f}")

#4. 마무리
train_data.shape #(901, 1850)
pca_train_data.shape #(901, 72)

print(f"Baseline test accuracy is {test_acc:.4f}")
print(f"Eigenface test accuracy is {pca_test_acc:.4f}")
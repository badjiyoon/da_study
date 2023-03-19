#CH07_04. 얼굴 사진 분류(python)

#SVM으로 얼굴 사진 분류하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
"sklearn.datasets의 fetch_lfw_people"
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

data, target = faces["data"], faces["target"]

#1.2 Data EDA
"이미지의 height와 width를 확인"
n_samples, h, w = faces.images.shape
n_samples, h, w

"얼굴의 주인들의 이름을 확인"
target_names = faces.target_names
n_classes = target_names.shape[0]
target_names

"이미지를 실제로 확인"
samples = data[:10].reshape(10, h, w)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    ax = axes[idx//5, idx%5]
    ax.imshow(sample, cmap="gray")
    ax.set_title(target_names[target[idx]])

#1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_target)}, {len(train_target)/len(data):.2f}")
print(f"test_data size: {len(test_target)}, {len(test_target)/len(data):.2f}")

#1.4 Data Scaling
"SVM 역시 거리를 기반으로 모델을 학습하기 때문에 데이터의 범위를 줄여주어야 함"
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_data)

scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

#2. SVM

#2.1 Baseline
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

#2.2 Hyper Parameter Tuning
from sklearn.model_selection import GridSearchCV

#2.2.1 탐색 범위 설정
params = [
    {"kernel": ["linear"], "C": [10, 30, 100, 300, 1000, 3000, 10000, 30000]},
    {
        "kernel": ["rbf"],
        "C": [1, 3, 10, 30, 100, 300, 1000],
        "gamma": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
    },
]
params

#2.2.2 탐색
grid_cv = GridSearchCV(svm, params, cv=3, n_jobs=-1)
grid_cv.fit(scaled_train_data, train_target)

#2.2.3 결과
print(f"Best score of paramter search is: {grid_cv.best_score_:.4f}")

print("Best parameter of best score is")
for key, value in grid_cv.best_params_.items():
    print(f"\t {key}: {value}")

train_pred = grid_cv.best_estimator_.predict(scaled_train_data)
test_pred = grid_cv.best_estimator_.predict(scaled_test_data)

best_train_acc = accuracy_score(train_target, train_pred)
best_test_acc = accuracy_score(test_target, test_pred)

print(f"Best Parameter train accuracy is {best_train_acc:.4f}")
print(f"Best Parameter test accuracy is {best_test_acc:.4f}")

#3. 마무리
print(f"Baseline test accuracy is {test_acc:.4f}")
print(f"Best Parameter test accuracy is {best_test_acc:.4f}")
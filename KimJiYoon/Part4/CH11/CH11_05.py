# 샘플 데이터와 Out-of-Distribution 모델
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
from sklearn.datasets import make_moons

data, label = make_moons(n_samples=300, shuffle=True, noise=0.5, random_state=2021)
plt.scatter(data[:, 0], data[:, 1], c=label)

# 1.2 Resample Data
from imblearn.datasets import make_imbalance
from collections import Counter


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return {minority_class: int(multiplier * target_stats[minority_class])}


data, label = make_imbalance(
    data,
    label,
    sampling_strategy=ratio_func,
    **{"multiplier": 0.1, "minority_class": 1, }
)

plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()

# 1.3 Split Data
# 학습시 정상데이터만 있어야함.
normal_data, abnormal_data = data[label == 0], data[label == 1]
normal_label, abnormal_label = label[label == 0], label[label == 1]
normal_label
abnormal_label

from sklearn.model_selection import train_test_split

train_data, test_normal_data, train_label, test_normal_label = train_test_split(
    normal_data, normal_label, train_size=0.7, random_state=2021
)

test_data = np.concatenate([test_normal_data, abnormal_data])
test_label = np.concatenate([test_normal_label, abnormal_label])

plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

test_label.mean()
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 1.4 시각화 데이터
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 2. Isolation Forest
from sklearn.ensemble import IsolationForest

isol_forest = IsolationForest()
# 2.1 학습 & 예측

isol_forest.fit(train_data, train_label)
isol_test_pred = isol_forest.predict(test_data)
isol_test_pred
# 1은 정상 -1는 비정상
isol_forest.decision_function(test_data)
# 2.2 평가
# 정확도
from sklearn.metrics import accuracy_score

isol_test_acc = accuracy_score(test_label, isol_test_pred == -1)
print(f"Isolation Forest Test Accuracy is {isol_test_acc:.4f}")

# F1 Score
from sklearn.metrics import f1_score

isol_test_f1 = f1_score(test_label, isol_test_pred == -1)
print(f"Isolation Forest Test F1-Score is {isol_test_f1:.4f}")

# 2.3 시각화
isol_Z = isol_forest.predict(np.c_[xx.ravel(), yy.ravel()])
isol_Z = isol_Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, isol_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

cs = plt.contourf(xx, yy, isol_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 3. OCSVMß
# 3.1 학습 & 예측
from sklearn.svm import OneClassSVM

ocsvm = OneClassSVM()
ocsvm.fit(train_data, train_label)
ocsvm_test_pred = ocsvm.predict(test_data)
ocsvm_test_pred

# 3.2 평가
ocsvm_test_acc = accuracy_score(test_label, ocsvm_test_pred == -1)
print(f"OCSVM Test Accuracy is {ocsvm_test_acc:.4f}")

# F1 score
ocsvm_test_f1 = f1_score(test_label, ocsvm_test_pred == -1)
print(f"OCSVM Test F1-Score is {ocsvm_test_f1:.4f}")

# 3.3. 시각화
ocsvm_Z = ocsvm.predict(np.c_[xx.ravel(), yy.ravel()])
ocsvm_Z = ocsvm_Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, ocsvm_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

cs = plt.contourf(xx, yy, ocsvm_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 4. PCA
# 4.1 학습 & 예측

from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(train_data)
test_latent = pca.transform(test_data)
test_latent[:10]
test_recon = pca.inverse_transform(test_latent)
recon_diff = (test_data - test_recon) ** 2
test_data[0]
test_recon[0]
recon_diff[0]
pca_pred = recon_diff.mean(1)
pca_pred[:10]

# 4.2 평가
from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(test_label, pca_pred)
pca_auroc = auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.show()

print(f"PCA test AUROC is {pca_auroc:.4f}")

# Best Threshold
f1_scores = []
for t in threshold:
    pca_test_pred = pca_pred > t
    pca_test_f1 = f1_score(test_label, pca_test_pred)
    f1_scores += [pca_test_f1]
    print(f"threshold: {t:.4f}, f1-score: {pca_test_f1:.4f}")

best_thresh = threshold[np.argmax(f1_scores)]
best_thresh
pca_test_pred = pca_pred > best_thresh
pca_test_pred
# 정확도
pca_test_acc = accuracy_score(test_label, pca_test_pred)
print(f"PCA Test Accuracy is {pca_test_acc:.4f}")

# F1 Score
pca_test_f1 = f1_score(test_label, pca_test_pred)
print(f"PCA Test F1-Score is {pca_test_f1:.4f}")

# 4.3 시각화
Z = np.c_[xx.ravel(), yy.ravel()]
Z_latent = pca.transform(Z)
Z_recon = pca.inverse_transform(Z_latent)
pca_Z = (Z - Z_recon).mean(1)
pca_Z = list(map(int, pca_Z > best_thresh))
pca_Z = np.array(pca_Z).reshape(xx.shape)
cs = plt.contourf(xx, yy, pca_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

cs = plt.contourf(xx, yy, pca_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 5. 마무리
# 5.1 정확도
print(f"Isolation Forest Test Accuracy is {isol_test_acc:.4f}")
print(f"OCSVM Test Accuracy is {ocsvm_test_acc:.4f}")
print(f"PCA Test Accuracy is {pca_test_acc:.4f}")

# 5.2 F1-Score
print(f"Isolation Forest Test F1-Score is {isol_test_f1:.4f}")
print(f"OCSVM Test F1-Score is {ocsvm_test_f1:.4f}")
print(f"PCA Test F1-Score is {pca_test_f1:.4f}")

# 샘플 데이터와 샘플링 기법
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)
# 1. Data
# 1.1 Sample Data
from sklearn.datasets import make_moons

data, label = make_moons(n_samples=300, shuffle=True, noise=0.5, random_state=2021)
plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()

# 1.2 Resample Data

from imblearn.datasets import make_imbalance
from collections import Counter


def ratio_func(y, multiplier, minority_class):
    target_stats = Counter(y)
    return { minority_class: int(multiplier * target_stats[minority_class]) }


data, label = make_imbalance(
    data,
    label,
    sampling_strategy=ratio_func,
    **{"multiplier": 0.1, "minority_class": 1, }
)

plt.scatter(data[:, 0], data[:, 1], c=label)
plt.show()

pd.Series(label).value_counts()

# 1.3 Split Data
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021, stratify=label
)

train_label.mean()

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

# 2. Model
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()
# 2.1 학습 & 예측
tree.fit(train_data, train_label)

tree_train_pred = tree.predict(train_data)
tree_test_pred = tree.predict(test_data)

# 2.3 평가
# 정확도
from sklearn.metrics import accuracy_score

tree_train_acc = accuracy_score(train_label, tree_train_pred)
tree_test_acc = accuracy_score(test_label, tree_test_pred)

print(f"Tree train accuray is {tree_train_acc:.4f}")
print(f"Tree test accuray is {tree_test_acc:.4f}")

# F1 Score
from sklearn.metrics import f1_score

tree_train_f1 = f1_score(train_label, tree_train_pred)
tree_test_f1 = f1_score(test_label, tree_test_pred)

print(f"Tree train F1-Score is {tree_train_f1:.4f}")
print(f"Tree test F1-Score is {tree_test_f1:.4f}")

# 2.4 시각화
tree_Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
tree_Z = tree_Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, tree_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

cs = plt.contourf(xx, yy, tree_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 3. Under Sampling
# 3.1 Under Sampling Data
from imblearn.under_sampling import RandomUnderSampler, NearMiss

under_dict = {}

rus = RandomUnderSampler(random_state=2021)
rus_data, rus_label = rus.fit_resample(train_data, train_label)
under_dict["rus"] = {"data": rus_data, "label": rus_label}

for i in range(1, 4):
    near_miss = NearMiss(version=i)
    near_data, near_label = near_miss.fit_resample(train_data, train_label)
    under_dict[f"near_{i}"] = {
        "data": near_data, "label": near_label
    }

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for idx, (name, sample) in enumerate(under_dict.items()):
    ax = axes[idx // 2, idx % 2]
    d, l = sample["data"], sample["label"]
    ax.scatter(d[:, 0], d[:, 1], c=l)
    ax.set_title(name)
plt.show()

# 3.2 학습
under_model = {}
for name, sample in under_dict.items():
    under_tree = DecisionTreeClassifier()
    under_tree.fit(sample["data"], sample["label"])
    under_model[name] = under_tree

under_model

# 3.3 예측
under_pred = {}
for name, under_tree in under_model.items():
    under_test_pred = under_tree.predict(test_data)
    under_pred[name] = under_test_pred

under_pred

# 3.4 평가
# 정확도
for name, pred in under_pred.items():
    acc = accuracy_score(test_label, pred)
    print(f"{name} Sampling test accuray is {acc:.4f}")

for name, pred in under_pred.items():
    f1 = f1_score(test_label, pred)
    print(f"{name} Sampling test F1-Score is {f1:.4f}")

# 3.5 시각화
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for idx, (name, under_tree) in enumerate(under_model.items()):
    ax = axes[idx // 2, idx % 2]
    under_Z = under_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    under_Z = under_Z.reshape(xx.shape)
    ax.contourf(xx, yy, under_Z, cmap=plt.cm.Paired)
    ax.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
    ax.set_title(name)
plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
for idx, (name, under_tree) in enumerate(under_model.items()):
    ax = axes[idx // 2, idx % 2]
    under_Z = under_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    under_Z = under_Z.reshape(xx.shape)
    ax.contourf(xx, yy, under_Z, cmap=plt.cm.Paired)
    ax.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
    ax.set_title(name)
plt.show()

# 4. Over Sampling
# 4.1 Over Sampling Data
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=2021)
smote_data, smote_label = smote.fit_resample(train_data, train_label)
smote_data[:10]
smote_label[:10]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

axes[0].scatter(train_data[:, 0], train_data[:, 1], c=train_label)
axes[0].set_title("raw data")

axes[1].scatter(smote_data[:, 0], smote_data[:, 1], c=smote_label)
axes[1].set_title("smote data")
plt.show()

# 4.2 학습
smote_tree = DecisionTreeClassifier()
smote_tree.fit(smote_data, smote_label)

# 4.2 예측
smote_test_pred = smote_tree.predict(test_data)

# 4.3 평가
smote_acc = accuracy_score(test_label, smote_test_pred)
print(f"SMOTE test accuray is {smote_acc:.4f}")

smote_f1 = f1_score(test_label, smote_test_pred)
print(f"SMOTE test F1-Score is {smote_f1:.4f}")

smote_Z = smote_tree.predict(np.c_[xx.ravel(), yy.ravel()])
smote_Z = smote_Z.reshape(xx.shape)

cs = plt.contourf(xx, yy, smote_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:, 0], train_data[:, 1], c=train_label)
plt.show()

cs = plt.contourf(xx, yy, smote_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_label)
plt.show()

# 5. 마무리
# 5.1 정확도
print(f"Tree test accuray is {tree_test_acc:.4f}")
for name, pred in under_pred.items():
    acc = accuracy_score(test_label, pred)
    print(f"{name} Sampling test accuray is {acc:.4f}")
print(f"SMOTE test accuray is {smote_acc:.4f}")

# 5.2 F1 Score
print(f"Tree test F1-Score is {tree_test_f1:.4f}")
for name, pred in under_pred.items():
    f1 = f1_score(test_label, pred)
    print(f"{name} Sampling test F1-Score is {f1:.4f}")
print(f"SMOTE test F1-Score is {smote_f1:.4f}")

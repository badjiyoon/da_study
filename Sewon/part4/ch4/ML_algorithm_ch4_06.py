#CH04_06. Iris 꽃 종류 분류 (Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
#데이터는 sklearn.datasets 의 load_iris 함수를 이용해 받을 수 있습니다.
from sklearn.datasets import load_iris

iris = load_iris()

#데이터에서 사용되는 변수는 암술과 수술의 길이와 넓이입니다.
iris["feature_names"]

#정답은 iris 꽃의 종류입니다.
iris["target_names"]
data, target = iris["data"], iris["target"]

#1.2 데이터 EDA
pd.DataFrame(data, columns=iris["feature_names"]).describe()

#1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021, stratify=target
)

print("train data 개수:", len(train_data))
print("train data 개수:", len(test_data))

#1.4 시각화

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))

pair_combs = [
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
]
for idx, pair in enumerate(pair_combs):
    x, y = pair
    ax = axes[idx//3, idx%3]
    ax.scatter(
        x=train_data[:, x], y=train_data[:, y], c=train_target, edgecolor='black', s=15
    )
    ax.set_xlabel(iris["feature_names"][x])
    ax.set_ylabel(iris["feature_names"][y])
    
#2. Decision Tree

from sklearn.tree import DecisionTreeClassifier, plot_tree

gini_tree = DecisionTreeClassifier()

#2.1 학습
gini_tree.fit(train_data, train_target)

plt.figure(figsize=(10,10))
plot_tree(gini_tree, feature_names=iris["feature_names"], class_names=iris["target_names"])

#2.2 Arguments
"""
DecisionTreeClassifier에서 주로 탐색하는 argument들은 다음과 같습니다.

criterion: 어떤 정보 이득을 기준으로 데이터를 나눌지 정합니다. (gini, entropy)
max_depth: 나무의 최대 깊이를 정해줍니다.
min_samples_split: 노드가 나눠질 수 있는 최소 데이터 개수를 정합니다.
"""

#2.2.1 max_depth
depth_1_tree = DecisionTreeClassifier(max_depth=1)
depth_1_tree.fit(train_data, train_target)

plot_tree(depth_1_tree, feature_names=iris["feature_names"], class_names=iris["target_names"])

#2.2.2 min_samples_split

sample_50_tree = DecisionTreeClassifier(min_samples_split=50)
sample_50_tree.fit(train_data, train_target)

plot_tree(sample_50_tree, feature_names=iris["feature_names"], class_names=iris["target_names"])

#2.2.3 criterion
entropy_tree = DecisionTreeClassifier(criterion="entropy")
entropy_tree.fit(train_data, train_target)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))
plot_tree(gini_tree, feature_names=iris["feature_names"], class_names=iris["target_names"], ax=axes[0])
plot_tree(entropy_tree, feature_names=iris["feature_names"], class_names=iris["target_names"], ax=axes[1])
plt.show()

#2.3 예측
trees = [
    ("gini tree", gini_tree),
    ("entropy tree", entropy_tree),
    ("depth=1 tree", depth_1_tree),
    ("sample=50 tree" ,sample_50_tree),
]

train_preds = []
test_preds = []
for tree_name, tree in trees:
    train_pred = tree.predict(train_data)
    test_pred =  tree.predict(test_data)
    train_preds += [train_pred]
    test_preds += [test_pred]

train_preds

#2.3 평가하기
from sklearn.metrics import accuracy_score

for idx, (tree_name, tree) in enumerate(trees):
    train_acc = accuracy_score(train_target, train_preds[idx])
    test_acc =  accuracy_score(test_target, test_preds[idx])
    print(tree_name)
    print("\t", f"train accuracy is {train_acc:.2f}")
    print("\t", f"test accuracy is {test_acc:.2f}")

#2.4 Feature Importance
iris["feature_names"]
gini_tree.feature_importances_
gini_feature_importance = pd.Series(gini_tree.feature_importances_, index=iris["feature_names"])
gini_feature_importance

gini_feature_importance.plot(kind="barh", title="gini tree feature importance")

sample_50_feature_importance = pd.Series(
    sample_50_tree.feature_importances_,
    index=iris["feature_names"]
)

sample_50_feature_importance.plot(kind="barh", title="sample=50 tree feature importance")

#3. 시각화
def plot_decision_boundary(pair_data, pair_tree, ax):
    x_min, x_max = pair_data[:, 0].min() - 1, pair_data[:, 0].max() + 1
    y_min, y_max = pair_data[:, 1].min() - 1, pair_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    Z = pair_tree.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = ax.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    # Plot the training points
    for i, color in zip(range(3), "ryb"):
        idx = np.where(train_target == i)
        ax.scatter(pair_data[idx, 0], pair_data[idx, 1], c=color, label=iris["target_names"][i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=15)
    return ax

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15,10))

pair_combs = [
    [0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]
]
for idx, pair in enumerate(pair_combs):
    x, y = pair
    pair_data = train_data[:, pair]
    pair_tree = DecisionTreeClassifier().fit(pair_data, train_target)

    ax = axes[idx//3, idx%3]
    ax = plot_decision_boundary(pair_data, pair_tree, ax)
    ax.set_xlabel(iris["feature_names"][x])
    ax.set_ylabel(iris["feature_names"][y])


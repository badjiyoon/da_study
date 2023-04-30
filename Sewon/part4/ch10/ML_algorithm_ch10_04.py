#CH10_04. Boositng Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Sample Data
rom sklearn.datasets import make_gaussian_quantiles


data_1, label_1 = make_gaussian_quantiles(
    cov=2, n_samples=200, n_features=2, n_classes=2, random_state=2021
)
data_2, label_2 = make_gaussian_quantiles(
    mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=2021
)

data = np.concatenate((data_1, data_2))
label = np.concatenate((label_1, - label_2 + 1))

plt.scatter(data[:,0], data[:,1], c=label)
plt.show()

#1.2 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)

#1.3 시각화 데이터
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
plt.show()

#2. Decision Tree
rom sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(max_depth=2)

#2.1 학습
tree.fit(train_data, train_label)

#2.2 예측
tree_train_pred = tree.predict(train_data)
tree_test_pred = tree.predict(test_data)

#2.3 평가
from sklearn.metrics import accuracy_score

tree_train_acc = accuracy_score(train_label, tree_train_pred)
tree_test_acc = accuracy_score(test_label, tree_test_pred)

print(f"Tree train accuray is {tree_train_acc:.4f}")
print(f"Tree test accuray is {tree_test_acc:.4f}")   

#2.4 시각화
tree_Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
tree_Z = tree_Z.reshape(xx.shape)

plt.figure(figsize=(14, 7))
plt.subplot(121)
cs = plt.contourf(xx, yy, tree_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:,0], train_data[:,1], c=train_label)
plt.title("train data")

plt.subplot(122)
cs = plt.contourf(xx, yy, tree_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:,0], test_data[:,1], c=test_label)
plt.title("test data")
plt.show()

#3. AdaBoost
"""
AdaBoost는 sklearn.ensemble의 AdaBoostClassifier로 생성
AdaBoostClassifier는 base_estimator를 선언
가장 간단한 if else로 데이터가 분류 될 수 있도록 depth가 1인 tree로 base estimator 생성
"""
from sklearn.ensemble import AdaBoostClassifier

ada_boost = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))

#3.1 학습
ada_boost.fit(train_data, train_label)

#3.2 예측
ada_boost_train_pred = ada_boost.predict(train_data)
ada_boost_test_pred = ada_boost.predict(test_data)

#3.3 평가
from sklearn.metrics import accuracy_score

ada_boost_train_acc = accuracy_score(train_label, ada_boost_train_pred)
ada_boost_test_acc = accuracy_score(test_label, ada_boost_test_pred)

print(f"Ada Boost train accuray is {ada_boost_train_acc:.4f}")
print(f"Ada Boost test accuray is {ada_boost_test_acc:.4f}")   

#3.4 시각화
ada_boost_Z = ada_boost.predict(np.c_[xx.ravel(), yy.ravel()])
ada_boost_Z = ada_boost_Z.reshape(xx.shape)

plt.figure(figsize=(14, 7))

plt.subplot(121)
cs = plt.contourf(xx, yy, ada_boost_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:,0], train_data[:,1], c=train_label)
plt.title("train_data")

plt.subplot(122)
cs = plt.contourf(xx, yy, ada_boost_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:,0], test_data[:,1], c=test_label)
plt.title("test_data")

#4. GradientBoost
"""
Gradient Boost는 sklearn.ensemble 의 GradientBoostingClassifier로 생성
Gradient Boost또한 간단한 if else로 만들 수 있도록 max_depth를 1로 주겠습니다.
"""
from sklearn.ensemble import GradientBoostingClassifier

grad_boost = GradientBoostingClassifier(max_depth=1)

#4.1 학습
grad_boost.fit(train_data, train_label)

#4.2 예측
grad_boost_train_pred = grad_boost.predict(train_data)
grad_boost_test_pred = grad_boost.predict(test_data)

#4.3 평가
from sklearn.metrics import accuracy_score

grad_boost_train_acc = accuracy_score(train_label, grad_boost_train_pred)
grad_boost_test_acc = accuracy_score(test_label, grad_boost_test_pred)

print(f"Gradient Boost train accuray is {grad_boost_train_acc:.4f}")
print(f"Gradient Boost test accuray is {grad_boost_test_acc:.4f}") 

#4.4 시각화
grad_boost_Z = grad_boost.predict(np.c_[xx.ravel(), yy.ravel()])
grad_boost_Z = grad_boost_Z.reshape(xx.shape)

plt.figure(figsize=(14, 7))

plt.subplot(121)
cs = plt.contourf(xx, yy, grad_boost_Z, cmap=plt.cm.Paired)
plt.scatter(train_data[:,0], train_data[:,1], c=train_label)
plt.title("train_data")

plt.subplot(122)
cs = plt.contourf(xx, yy, grad_boost_Z, cmap=plt.cm.Paired)
plt.scatter(test_data[:,0], test_data[:,1], c=test_label)
plt.title("test_data")
plt.show()

#5. 마무리
print(f"Tree test accuray is {tree_test_acc:.4f}")
print(f"Gradient Boost test accuray is {grad_boost_test_acc:.4f}")
print(f"Ada Boost test accuray is {ada_boost_test_acc:.4f}")

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
Z_name = [
    ("tree", tree_Z),
    ("Ada Boost", ada_boost_Z),
    ("Gradient Boost", grad_boost_Z)
]
for idx, (name, Z) in enumerate(Z_name):
    ax = axes[idx]
    ax.contourf(xx, yy, Z, cmap=plt.cm.Paired)
    ax.scatter(train_data[:,0], train_data[:,1], c=train_label)
    ax.set_title(name)
plt.show()
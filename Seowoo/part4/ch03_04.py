import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(2021)

## 1. Data

# 1.1 Data Load
iris = load_iris()

print(iris["feature_names"])
print(iris["target_names"])

data, target = iris["data"], iris["target"]

# 1.2 Data EDA
# Data 분포
print(pd.DataFrame(data, columns=iris["feature_names"]).describe())
# 종류별 개수
print(pd.Series(target).value_counts())

# 1.3 Data Split
train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print("train data 갯수 : ", len(train_data))
print("test data 갯수 : ", len(test_data))

# Train 데이터 갯수
print(pd.Series(train_target).value_counts())
# Test 데이터 갯수
print(pd.Series(test_target).value_counts())

##### 단순히 데이터를 분류할 경우 target 분포를 반영 하지 못 하므로 stratify 옵션을 넣어준다.
##### 각 label에 대한 데이터를 골고루 분포 시키기 위한 작업
train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2121, stratify=target
)
# Train 데이터 갯수
print(pd.Series(train_target).value_counts())
# Test 데이터 갯수
print(pd.Series(test_target).value_counts())

## 2. Multiclass
X = train_data[:, :2]

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=train_target, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.show()

# 2.1 One vs Rest
ovr_logit = LogisticRegression(multi_class='ovr')
ovr_logit.fit(X, train_target)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=ovr_logit.predict(X), edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal length')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

coef = ovr_logit.coef_
intercept = ovr_logit.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

    plt.plot([x_min, x_max], [line(x_min), line(x_max)], ls='--', color=color)


for i, color in zip(ovr_logit.classes_, 'bry'):
    plot_hyperplane(i, color)
plt.show()

# 2.2 Multinomial
multi_logit = LogisticRegression(multi_class='multinomial')
multi_logit.fit(X, train_target)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=multi_logit.predict(X), edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal length')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

coef = multi_logit.coef_
intercept = multi_logit.intercept_

for i, color in zip(multi_logit.classes_, 'bry'):
    plot_hyperplane(i, color)
plt.show()

## 3. Logistic Regression (Multinomial)
multi_logit = LogisticRegression()

multi_logit.fit(train_data, train_target)

train_pred_proba = multi_logit.predict_proba(train_data)
sample_pred = train_pred_proba[0]
print(sample_pred)

print(f"class 0에 속하지 않을 확률 : {1 - sample_pred[0]:.4f}")
print(f"class 1과 2에 속할 확률 : {sample_pred[1:].sum():.4f}")

train_pred = multi_logit.predict(train_data)
test_pred = multi_logit.predict(test_data)

train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

print(f"Train accuracy is : {train_acc:.2f}")
print(f"Test accuracy is : {test_acc:.2f}")

## 4. Logistic Regression (OVR)
ovr_logit = LogisticRegression(multi_class='ovr')

# 4.1 학습
ovr_logit.fit(train_data, train_target)

# 4.2 예측
ovr_train_pred = ovr_logit.predict(train_data)
ovr_test_pred = ovr_logit.predict(test_data)

ovr_train_acc = accuracy_score(train_target, ovr_train_pred)
ovr_test_acc = accuracy_score(test_target, ovr_test_pred)

print(f"One vs Rest Train accuracy is : {ovr_train_acc:.2f}")
print(f"One vs Rest Test accuracy is : {ovr_test_acc:.2f}")

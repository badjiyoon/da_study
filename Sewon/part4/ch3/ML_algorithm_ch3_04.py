#CH03_04. Iris의 종류 분류(Multiclass)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
from sklearn.datasets import load_iris

iris = load_iris()
iris["feature_names"]
'''
변수명
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
'''

iris["target_names"]
data, target = iris["data"], iris["target"]
target

#1.2 데이터 EDA
pd.DataFrame(data, columns=iris["feature_names"]).describe()

pd.Series(target).value_counts()

#1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021)

print("train data 개수:", len(train_data))
print("train data 개수:", len(test_data))

pd.Series(train_target).value_counts() #Train data의 정답 개수
pd.Series(test_target).value_counts() #Test data의 정답 개수

'''
단순히 데이터를 분류할 경우 원래 데이터의 target 분포를 반영하지 못합니다.
이때 사용하는 것이 startify 옵션입니다.
이 옵션에 데이터의 label을 넣어주면 원본 데이터의 정답 분포를 반영해 데이터를 나눠줍니다.
'''

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021, stratify=target) #startify에 target label 추가

pd.Series(train_target).value_counts()
pd.Series(test_target).value_counts()

#2. Multiclass
from sklearn.linear_model import LogisticRegression

X = train_data[:, :2]
X[0]

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=train_target, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(X[:,0].min()-0.5, X[:,0].max()+0.5)
plt.ylim(X[:,1].min()-0.5, X[:,1].max()+0.5)
plt.show()

#2.1 One vs Rest
ovr_logit = LogisticRegression(multi_class="ovr")
ovr_logit.fit(X, train_target)

x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

plt.figure(1, figsize=(10, 10))

plt.scatter(X[:, 0], X[:, 1], c=ovr_logit.predict(X), edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

coef = ovr_logit.coef_
intercept = ovr_logit.intercept_

def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    plt.plot([x_min, x_max], [line(x_min), line(x_max)],
             ls="--", color=color)

for i, color in zip(ovr_logit.classes_, "bry"):
    plot_hyperplane(i, color)

#2.2 Multinomial
'''
정답의 분포가 Multinomial 분포를 따른다고 가정한 후 시행하는 Multiclass Logistic Regression 입니다.
LogisticRegression의 기본 값은 "multinomial" 입니다.
'''

multi_logit = LogisticRegression(multi_class="multinomial")
multi_logit.fit(X, train_target)

x_min, x_max = X[:,0].min() - 0.5, X[:,0].max() + 0.5
y_min, y_max = X[:,1].min() - 0.5, X[:,1].max() + 0.5

plt.figure(1, figsize=(10, 10))

plt.scatter(X[:, 0], X[:, 1], c=multi_logit.predict(X), edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

coef = multi_logit.coef_
intercept = multi_logit.intercept_

def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    plt.plot([x_min, x_max], [line(x_min), line(x_max)],
             ls="--", color=color)

for i, color in zip(multi_logit.classes_, "bry"):
    plot_hyperplane(i, color)

plt.show()

#3. Logistic Regression (Multinomial)
multi_logit = LogisticRegression()

#3.1 학습
multi_logit.fit(train_data, train_target)

#3.2 예측
train_pred_proba = multi_logit.predict_proba(train_data)

sample_pred = train_pred_proba[0]
sample_pred

print(f"class 0에 속하지 않을 확률: {1 - sample_pred[0]:.4f}")
print(f"class 1과 2에 속할 확률: {sample_pred[1:].sum():.4f}")

train_pred = multi_logit.predict(train_data)
test_pred = multi_logit.predict(test_data)

#3.3 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

print(f"Train accuracy is : {train_acc:.2f}")
print(f"Test accuracy is : {test_acc:.2f}")

#4. Logistic Regression (OVR)
ovr_logit = LogisticRegression(multi_class="ovr")

#4.1 학습
ovr_logit.fit(train_data, train_target)

#4.2 예측
ovr_train_pred = ovr_logit.predict(train_data)
ovr_test_pred = ovr_logit.predict(test_data)

#4.3 평가
from sklearn.metrics import accuracy_score

ovr_train_acc = accuracy_score(train_target, ovr_train_pred)
ovr_test_acc = accuracy_score(test_target, ovr_test_pred)

print(f"One vs Rest Train accuracy is : {ovr_train_acc:.2f}")
print(f"One vs Rest Test accuracy is : {ovr_test_acc:.2f}")
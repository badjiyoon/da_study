import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# Data
# 1.1 Data Load
from sklearn.datasets import load_iris

iris = load_iris()
print(iris['feature_names'])
print(iris['target_names'])
data, target = iris['data'], iris['target']
print(target)

# Data EDA
print(pd.DataFrame(data, columns=iris['feature_names']).describe())
print(pd.Series(target).value_counts())

# Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print('Train data 개수 : ', len(train_data))
print('Test data 개수 : ', len(test_data))
print(pd.Series(train_target).value_counts())
print(pd.Series(test_target).value_counts())

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021, stratify=target
)

print(pd.Series(train_target).value_counts())
print(pd.Series(test_target).value_counts())

# Multiclass
from sklearn.linear_model import LogisticRegression

X = train_data[:, :2]
print(X[0])

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=train_target, edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.show()

# One vs Rest
ovr_logit = LogisticRegression(multi_class='ovr')
ovr_logit.fit(X, train_target)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

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
            ls='--', color=color)


for i, color in zip(ovr_logit.classes_, 'bry'):
    plot_hyperplane(i, color)

plt.show()

# Multinomial
multi_logic = LogisticRegression(multi_class='multinomial')
multi_logic.fit(X, train_target)

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.figure(1, figsize=(10, 10))
plt.scatter(X[:, 0], X[:, 1], c=multi_logic.predict(X), edgecolors='k', cmap=plt.cm.Paired)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

coef = multi_logic.coef_
intercept = multi_logic.intercept_


def plot_hyperplane(c, color):
    def line(x0):
        return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]
    plt.plot([x_min, x_max], [line(x_min), line(x_max)],
            ls='--', color=color)


for i, color in zip(multi_logic.classes_, 'bry'):
    plot_hyperplane(i, color)

plt.show()

# Logistice Regression (multinomial)
multi_logic = LogisticRegression()

multi_logic.fit(train_data, train_target)

train_pred_proba = multi_logic.predict_proba(train_data)

sample_pred = train_pred_proba[0]
print(sample_pred)

print(f'class 0에 속하지 않을 확률: {1 - sample_pred[0]:.4f}')
print(f'class 1과 2에 속할 확률: {sample_pred[1:].sum():.4f}')

train_pred = multi_logic.predict(train_data)
test_pred = multi_logic.predict(test_data)

from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

print(f'Train accuracy is : {train_acc:.2f}')
print(f'Test accuracy is : {test_acc:.2f}')


# 4. Logistic Regression(OVR)
ovr_logit = LogisticRegression(multi_class='ovr')
ovr_logit.fit(train_data, train_target)

ovr_train_pred = ovr_logit.predict(train_data)
ovr_test_pred = ovr_logit.predict(test_data)

ovr_train_acc = accuracy_score(train_target, ovr_train_pred)
ovr_test_acc = accuracy_score(test_target, ovr_test_pred)

print(f'One vs Rest Train accuracy is : {ovr_train_acc:.2f}')
print(f'One vs Rest Test accuracy is : {ovr_test_acc:.2f}')
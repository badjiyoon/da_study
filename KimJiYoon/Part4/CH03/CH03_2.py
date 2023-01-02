import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

np.random.seed(2021)

# 1. Data
# 1.1 Data Load
cancer = load_breast_cancer()
print(cancer['feature_names'])
# 정답
# malignant
# bengin
print(cancer['target_names'])
data, target = cancer["data"], cancer["target"]
print(data[0])
print(target[0])
# 1.2 Data DEA
df = pd.DataFrame(data, columns=cancer["feature_names"])
print(df.describe())
# 양성과 음성의 비율
print(pd.Series(target).value_counts())
# Histogram으로 그린다
plt.hist(target)

# mean radius와 정답간ㅇ늬 상관관계를 Plot으로 그려본다.
# mean radius가 클 경우 음성인 것을확인
plt.scatter(x=data[:, 0], y=target)
plt.xlabel('mean radius')
plt.ylabel("target")
plt.show()

# Data split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021,
)
print('train data 개수 : ', len(train_data))
print('test data 개수 : ', len(test_data))

# Linear Regression and Categorical Label
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()

# 학습
linear_regressor.fit(train_data, train_target)

# 예측
train_pred = linear_regressor.predict(train_data)
test_pred = linear_regressor.predict(test_data)

print(train_pred[:10])

# 시각화
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

preds = [
    ('Train', train_data, train_pred),
    ('Test', test_data, test_pred),
]

for idx, (name, d, pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(x=d[:, 0], y=pred)
    ax.axhline(0, color='red', linestyle='--')
    ax.axhline(1, color='red', linestyle='--')
    ax.set_xlabel('mean_radius')
    ax.set_ylabel('predict')
    ax.set_title(f'{name} Data')

# 평가하기
from sklearn.metrics import auc, roc_curve

fpr, tpr, threshold = roc_curve(train_target, train_pred)
auroc = auc(fpr, tpr)

print(fpr)
print(tpr)
print(threshold)

plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

print(f'AUROC : {auroc: .4f}')

# 이제 BEST Threshold를 계산해 보겠습니다.
J = tpr - fpr
idx = np.argmax(J)
best_thresh = threshold[idx]
print(f'Best_Threshold is : {best_thresh: .4f}')
print(f'Best_Threshold`s sensitivity is : {tpr[idx]: .4f}')
print(f'Best_Threshold`s specificity is : {1-fpr[idx]: .4f}')
print(f'Best_Threshold`s J is : {J[idx]: .4f}')

# Best Threshold는 AUROC 그래프에서 직석이 가장 긴 곳

plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
plt.plot((fpr[idx], fpr[idx]), (fpr[idx], tpr[idx]), color='red', linestyle='--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

preds = [
    ('Train', train_data, train_pred),
    ('Test', test_data, test_pred),
]

for idx, (name, d, pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(x=d[:, 0], y=pred)
    ax.axhline(0, color='red', linestyle='--')
    ax.axhline(1, color='red', linestyle='--')
    ax.set_xlabel('mean_radius')
    ax.set_ylabel('predict')
    ax.set_title(f'{name} Data')
    ax.axhline(best_thresh, color='blue')

train_pred_label = list(map(int, (train_pred > best_thresh)))
test_pred_label = list(map(int, (test_pred > best_thresh)))

from sklearn.metrics import accuracy_score
linear_train_accuracy = accuracy_score(train_target, train_pred_label)
linear_test_accuracy = accuracy_score(test_target, test_pred_label)

print(f'Train accuracy is : {linear_train_accuracy:.2f}')
print(f'Testaccuracy is : {linear_test_accuracy:.2f}')

plt.show()

# Logistic Regression
# Scaling
# Logistic Regression은 학습하기에 앞서 정규화 필요
# exp가 있는데 EXp는 값이 클 경우 overflow가 일언 날 수 있음
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_data)
# 학습된 scaler로 train/test 데이터를 변환합니다.
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

print(train_data[0])
print(scaled_train_data[0])

# 학습
from sklearn.linear_model import LogisticRegression
logit_regressor = LogisticRegression()
logit_regressor.fit(scaled_train_data, train_target)
# 에측
train_pred = logit_regressor.predict(scaled_train_data)
test_pred = logit_regressor.predict(scaled_test_data)
print(train_pred[:10])
train_pred_logit = logit_regressor.predict_proba(scaled_train_data)
test_pred_logit = logit_regressor.predict_proba(scaled_test_data)
print(train_pred_logit[:10])
print(train_pred_logit[0])
# 평가
train_pred_logit = train_pred_logit[:, 1]
test_pred_logit = test_pred_logit[:, 1]
print(train_pred_logit[0])

from sklearn.metrics import auc, roc_curve

fpr, tpr, threshold = roc_curve(train_target, train_pred)
auroc = auc(fpr, tpr)

print(fpr)
print(tpr)
print(threshold)

plt.plot(fpr, tpr)
plt.xlabel('fpr')
plt.ylabel('tpr')

print(f'AUROC : {auroc: .4f}')

# 이제 BEST Threshold를 계산해 보겠습니다.
J = tpr - fpr
idx = np.argmax(J)
best_thresh = threshold[idx]
print(f'Best_Threshold is : {best_thresh: .4f}')
print(f'Best_Threshold`s sensitivity is : {tpr[idx]: .4f}')
print(f'Best_Threshold`s specificity is : {1-fpr[idx]: .4f}')
print(f'Best_Threshold`s J is : {J[idx]: .4f}')

# Best Threshold는 AUROC 그래프에서 직석이 가장 긴 곳

plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
plt.plot((fpr[idx], fpr[idx]), (fpr[idx], tpr[idx]), color='red', linestyle='--')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.show()

plt.scatter(x=scaled_train_data[:, 0], y=train_pred_logit)
plt.axhline(best_thresh, color='blue')
plt.axhline(0, color='red', linestyle='--')
plt.axhline(1, color='red', linestyle='--')
plt.xlabel('mean radius')
plt.ylabel('Probabilty')
plt.show()

train_pred_label = list(map(int, (train_pred > best_thresh)))
test_pred_label = list(map(int, (test_pred > best_thresh)))

proba_train_accuracy = accuracy_score(train_target, train_pred_label)
proba_test_accuracy = accuracy_score(test_target, test_pred_label)

print(f'Train accuracy is : {proba_train_accuracy:.2f}')
print(f'Test accuracy is : {proba_test_accuracy:.2f}')

# predict의 결과값으로 정확도를 올림
train_accuracy = accuracy_score(train_target, train_pred)
test_accuracy = accuracy_score(test_target, test_pred)
print(f'Train accuracy is : {train_accuracy:.2f}')
print(f'Test accuracy is : {test_accuracy:.2f}')




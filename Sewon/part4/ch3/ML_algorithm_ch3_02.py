#CH03_02. 폐렴의 양성, 음성 분류

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data load
from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

type(cancer)
cancer["feature_names"]

'''
변수명
radius (mean of distances from center to points on the perimeter)
texture (standard deviation of gray-scale values)
perimeter
area
smoothness (local variation in radius lengths)
compactness (perimeter^2 / area - 1.0)
concavity (severity of concave portions of the contour)
concave points (number of concave portions of the contour)
symmetry
fractal dimension ("coastline approximation" - 1)
'''

cancer["target_names"]

'''
malignant
benign
0은 양성, 1은 음성
'''

data, target = cancer["data"], cancer["target"]
data[0] #데이터 확인
target[0] #정답 확인

#1.2 Data EDA
df = pd.DataFrame(data, columns=cancer["feature_names"])
df.describe()

data.shape #데이터 크기 확인

pd.Series(target).value_counts() #양성과 음성 비율 확인
plt.hist(target) #Histogram 확인
plt.show()

plt.scatter(x=data[:,0], y=target)
plt.xlabel("mean radius")
plt.ylabel("target")
plt.show()

#1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021)

print("train data 개수:", len(train_data))
print("test data 개수:", len(test_data))


#2. Linear Regression and Categorical Label
#Linear Regression으로 학습한다면?

from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

#2.1 학습
linear_regressor.fit(train_data, train_target)

#2.2 예측
train_pred = linear_regressor.predict(train_data)
test_pred = linear_regressor.predict(test_data)

train_pred[:10] #0~1 사이를 벗어난 값이 많이 나타난다.

#2.3 시각화
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

preds = [("Train", train_data, train_pred),
    ("Test", test_data, test_pred)]

for idx, (name, d, pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(x=d[:,0], y=pred)
    ax.axhline(0, color="red", linestyle="--")
    ax.axhline(1, color="red", linestyle="--")
    ax.set_xlabel("mean_radius")
    ax.set_ylabel("predict")
    ax.set_title(f"{name} Data")
plt.show()

#2.4 평가하기
'''
Linear Regression의 성능을 측정하기 위해서는 우선 예측값을 0과 1로 변환
Youden's Index를 이용해 Best Threshold를 찾은 후 0과 1로 변화시킨 후 정확도 확인
'''

from sklearn.metrics import auc, roc_curve

fpr, tpr, threshold = roc_curve(train_target, train_pred) #roc_curve(정답, 예측값)
auroc = auc(fpr, tpr)

fpr
tpr
threshold

#AUROC 그리기
plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

print(f"AUROC : {auroc:.4f}") #중간에 f는 뭐지?

#Best Threshold 
# -> Youdens' Index, AUROC 그래프에서 직선이 가장 긴 곳
np.argmax(tpr - fpr) #몇 번째 값인지 출력

J = tpr - fpr
idx = np.argmax(J)
best_thresh = threshold[idx]
print(f"Best Threshold is {best_thresh:.4f}")
print(f"Best Threshold's sensitivity is {tpr[idx]:.4f}")
print(f"Best Threshold's specificity is {1-fpr[idx]:.4f}")
print(f"Best Threshold's J is {J[idx]:.4f}")

plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
plt.plot((fpr[idx], fpr[idx]), (fpr[idx], tpr[idx]), color="red", linestyle="--")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

preds = [
    ("Train", train_data, train_pred),
    ("Test", test_data, test_pred),
]
for idx, (name, d, pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(x=d[:,0], y=pred)
    ax.axhline(0, color="red", linestyle="--")
    ax.axhline(1, color="red", linestyle="--")
    ax.set_xlabel("mean_radius")
    ax.set_ylabel("predict")
    ax.set_title(f"{name} Data")
    ax.axhline(best_thresh, color="blue")
plt.show()

#Threshold로 예측값을 0,1로 변환 후 정확도
train_pred_label = list(map(int, (train_pred > best_thresh)))
test_pred_label = list(map(int, (test_pred > best_thresh)))

from sklearn.metrics import accuracy_score

linear_train_accuracy = accuracy_score(train_target, train_pred_label)
linear_test_accuracy = accuracy_score(test_target, test_pred_label)

print(f"Train accuracy is : {linear_train_accuracy:.2f}")
print(f"Test accuracy is : {linear_test_accuracy:.2f}")

#3. Logistic Regression
#3.1 Scaling
'''
Logistic Regression은 학습하기에 앞서 학습시킬 데이터를 정규화해야 합니다.
Logistic Regressiond에는 exp가 있는데, exp는 값이 클 경우 overflow가 일어날 수 있기 때문입니다.

정규화는 항상 train data를 이용해 학습하고 valid, test 데이터를 변환해야 합니다.
모든 데이터를 한번에 학습할 경우 본 적이 없는 valid data의 평균과 분산이 반영되고 이는 overfitting을 일으키는 원인이 됩니다.
'''
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(train_data)

#학습된 scaler로 train/ test 데이터를 변환
scaled_train_data = scaler.transform(train_data)
scaled_test_data = scaler.transform(test_data)

train_data[0]
scaled_train_data[0]

#3.2 학습
from sklearn.linear_model import LogisticRegression

logit_regressor = LogisticRegression()

logit_regressor.fit(scaled_train_data, train_target)

#3.3 예측
'''
Classification 모델의 경우 예측하는 방법
1) predict: 해당 데이터가 어떤 class로 분류할지 바로 알려줍니다.
2) predict_proba: 각 class에 속할 확률을 보여줍니다.
'''

train_pred = logit_regressor.predict(scaled_train_data)
test_pred = logit_regressor.predict(scaled_test_data)

train_pred[:10]

train_pred_logit = logit_regressor.predict_proba(scaled_train_data)
test_pred_logit = logit_regressor.predict_proba(scaled_test_data)

train_pred_logit[:10]

train_pred_logit[0]

#3.4 평가
'''
데이터의 AUROC를 계산하기 위해서는 1의 클래스로 분류될 확률 하나만 필요한데
우리가 갖고 있는 예측값은 0과 1로 분류될 확률을 모두 표시하므로 1에 속할 확률만 남김
'''

train_pred_logit = train_pred_logit[:, 1]
test_pred_logit = test_pred_logit[:, 1]

train_pred_logit[0]

from sklearn.metrics import auc, roc_curve

fpr, tpr, threshold = roc_curve(train_target, train_pred_logit)
auroc = auc(fpr, tpr)

plt.plot(fpr, tpr)
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

print(f"AUROC : {auroc:.4f}")

#Best Threshold
J = tpr - fpr
idx = np.argmax(J)
best_thresh = threshold[idx]

print(f"Best Threshold is {best_thresh:.4f}")
print(f"Best Threshold's sensitivity is {tpr[idx]:.4f}")
print(f"Best Threshold's specificity is {1-fpr[idx]:.4f}")
print(f"Best Threshold's J is {J[idx]:.4f}")

plt.plot(fpr, tpr)
plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
plt.plot((fpr[idx],fpr[idx]), (fpr[idx], tpr[idx]), color="red", linestyle="--")
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.show()

plt.scatter(x=scaled_train_data[:,0], y=train_pred_logit)
plt.axhline(best_thresh, color="blue")
plt.axhline(0, color="red", linestyle="--")
plt.axhline(1, color="red", linestyle="--")
plt.xlabel("mean radius")
plt.ylabel("Probability")
plt.show()

#Threshold로 예측값을 0,1로 변환 후 정확도
train_pred_label = list(map(int, (train_pred_logit > best_thresh)))
test_pred_label = list(map(int, (test_pred_logit > best_thresh)))

proba_train_accuracy = accuracy_score(train_target, train_pred_label)
proba_test_accuracy = accuracy_score(test_target, test_pred_label)

print(f"Train accuracy is : {proba_train_accuracy:.2f}")
print(f"Test accuracy is : {proba_test_accuracy:.2f}")

#predict의 결과값으로 정확도
train_accuracy = accuracy_score(train_target, train_pred)
test_accuracy = accuracy_score(test_target, test_pred)

print(f"Train accuracy is : {train_accuracy:.2f}")
print(f"Test accuracy is : {test_accuracy:.2f}")

'''
predict_proba의 best_threshold로 계산한 결과와 predict로 계산한 결과가 다릅니다.
이는 두 0과 1로 예측하는 방법이 다르기 때문입니다.
예를 들어서 (0.49, 0.51)의 확률이 있을 때 predict의 경우 class 1의 확률에 속할 확률이 크기 때문에 1로 분류합니다.
하지만 best_threshold가 0.52라면 predict_proba의 경우 class를 0으로 분류하게 됩니다.
'''

#4. 마무리
print(f"Linear Regression Test Accuracy: {linear_test_accuracy:.2f}")
print(f"Logistic Regression predict_proba Test Accuracy: {proba_test_accuracy:.2f}")
print(f"Logistic Regression predict Test Accuracy: {test_accuracy:.2f}")
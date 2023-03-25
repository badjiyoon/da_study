import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# Simple Titanic Survival Data

# 1. Data
titanic = pd.read_csv("../../../comFiles/simple_titanic.csv")

# 데이터가 갖고 있는 column들은 다음과 같습니다.
# - Survived : 생존 유무 (정답)
# - Pclass : 탑승한 여객 클래스
# - Sex : 성별
# - Age : 나이
# - Fare : 운임료
# - Embarked: 승선항구 위치
#     - C = Cherbourg; Q = Queenstown; S = Southampton

titanic.columns

label = titanic["Survived"]
data = titanic.drop(["Survived"], axis=1)

# 1.1 Data EDA
# 데이터 통계치 확인
data.describe()
data["Sex"].value_counts()
data["Embarked"].value_counts()
# 빈 값 확인
data.isna().sum()
# 정답의 비율을 확인
label.value_counts()

# 1.2 Data Preprocess
data.loc[:, "Sex"] = data["Sex"].map({"male": 0, "female": 1})
data.loc[:, "Embarked"] = data["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# 1.3 Data Split
from sklearn.model_selection import train_test_split
train_data, validation_data, train_label, validation_label = train_test_split(data, label, train_size=0.6, random_state=2021)
valid_data, test_data, valid_label, test_label = train_test_split(validation_data, validation_label, train_size=0.5, random_state=2021)
print(f"train_data size: {len(train_label)}, {len(train_label)/len(data):.2f}")
print(f"valid_data size: {len(valid_label)}, {len(valid_label)/len(data):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(data):.2f}")
# 데이터 리셋
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data = test_data.reset_index(drop=True)
train_label = train_label.reset_index(drop=True)
valid_label = valid_label.reset_index(drop=True)
test_label = test_label.reset_index(drop=True)

# 2. 빈 데이터 채우기
# 빈 값이 많은 Age를 어떻게 채울까요?
# 1. 사용하지 않는다.
# 2. 전체 데이터의 평균으로 채운다.
# 3. 비슷한 데이터를 찾아서 비슷한 데이터의 평균으로 채운다.
na_cnt = data.isna().sum()
na_cnt.loc[na_cnt > 0].index

# 2.1 사용하지 않는 방법
# 데이터가 비어있는 row를 버리기 위해서는 DataFrame의 `dropna` 함수를 사용하면 됩니다.
# 하지만 이 방법은 Test데이터에 대해서 수행할 경우 비어있는 데이터를 처리할 수 있는 방법이 없어집니다.
drop_data = data.dropna()
print(f"전체 데이터 개수: {len(data)}")
print(f"값이 비어있는 데이터를 버린 후 데이터 개수: {len(drop_data)}")
print(f"버려진 데이터 개수: {len(data) - len(drop_data)}")

# 2.2 전체 데이터의 평균으로 채우는 방법
# 전체 데이터의 평균으로 빈 값을 채우는 방법은 쉽고 빠르게 사용할 수 있는 방법 중 하나입니다.
# `fillna` 함수를 이용해 빈 값을 쉽게 채울 수 있습니다.
mean_train_data = train_data.copy()
mean_valid_data = valid_data.copy()
mean_test_data = test_data.copy()
# 학습 데이터 Age의 평균
age_mean = mean_train_data["Age"].mean()
age_mean

# 비어있는 데이터들을 확인하고 값을 채움
mean_train_data.loc[:, "Age"] = mean_train_data["Age"].fillna(age_mean)
mean_valid_data.loc[:, "Age"] = mean_valid_data["Age"].fillna(age_mean)
mean_test_data.loc[:, "Age"] = mean_test_data["Age"].fillna(age_mean)

mean_train_data.isna().sum()

# 2.3 비슷한 데이터들의 평균으로 채우는 방법
# 이 방법은 빈 값이 있는 변수를 제거하고 나머지 변수들을 군집화하고 군집의 평균으로 빈 값을 채우는 방법입니다.
# 아이디어는 비슷한 데이터들은 변수 몇 개를 제거해도 같이 묶일 것이다에서 시작합니다.
cluster_train_data = train_data.copy()
cluster_valid_data = valid_data.copy()
cluster_test_data = test_data.copy()

# clustering 전에 데이터 정규화
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(cluster_train_data.drop(["Age"], axis=1))

train_fill_data = scaler.transform(cluster_train_data.drop(["Age"], axis=1))
valid_fill_data = scaler.transform(cluster_valid_data.drop(["Age"], axis=1))
test_fill_data = scaler.transform(cluster_test_data.drop(["Age"], axis=1))
# 최적의 K를 찾기 위해 K값에 따른 SSE 계산
from sklearn.cluster import KMeans

n_cluster = []
sse = []
for n in range(3, 15, 2):
    kmeans = KMeans(n_clusters=n)
    kmeans.fit(train_fill_data)
    n_cluster += [n]
    sse += [kmeans.inertia_]

plt.plot(n_cluster, sse)
plt.show()
# SSE 그래프에서 꺽이는 지점인 7로 K를 정함
n_clusters = 7

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(train_fill_data)

clustered_train = kmeans.predict(train_fill_data)
clustered_valid = kmeans.predict(valid_fill_data)
clustered_test = kmeans.predict(test_fill_data)
clustered_test

cluster_fill_value = {}
for i in range(n_clusters):
    class_mean = cluster_train_data.loc[clustered_train == i, "Age"].dropna().mean()
    cluster_fill_value[i] = class_mean

cluster_fill_value

# train data에서 빈 값을 채움
train_na_idx = cluster_train_data.loc[cluster_train_data["Age"].isna()].index
# 빈 값 데이터
train_na_idx
# 각 index가 속하는 군집은 다음과 같음
clustered_train[train_na_idx]
# 각 index에 채울 값들을 가져옴
train_fill_value = list(map(lambda x: cluster_fill_value[x], clustered_train[train_na_idx]))
train_fill_value[:10]

# 학습에 사용할 데이터에 빈 값을 채움
cluster_train_data.loc[train_na_idx, "Age"] = train_fill_value
# 빈 값이 모두 채워진 것을 확인할 수 있음
cluster_train_data.loc[train_na_idx]
cluster_train_data.head()
# Valid, Test 데이터에 대해서도 동일하게 진행합니다.
# Valid, Test 데이터의 빈 값을 채울 때에는 정규화와 동일하게 Train 데이터에서 구한 값으로 채워줍니다.
valid_na_idx = cluster_valid_data.loc[cluster_valid_data["Age"].isna()].index
valid_fill_value = list(map(lambda x: cluster_fill_value[x], clustered_valid[valid_na_idx]))

test_na_idx = cluster_test_data.loc[cluster_test_data["Age"].isna()].index
test_fill_value = list(map(lambda x: cluster_fill_value[x], clustered_test[test_na_idx]))

cluster_valid_data.loc[valid_na_idx, "Age"] = valid_fill_value
cluster_test_data.loc[test_na_idx, "Age"] = test_fill_value

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Hyper-Parameter Tuning
# 각 데이터에 대해서 최고의 성능을 보이는 `n_estimators`를 찾아 보겠습니다.
# 3.3.1 전체 데이터의 평균으로 채운 데이터

n_estimators = [n for n in range(50, 1050, 50)]

mean_accuracy = []
for n_estimator in n_estimators:
    mean_random_forest = RandomForestClassifier(n_estimators=n_estimator)
    mean_random_forest.fit(mean_train_data, train_label)
    mean_valid_pred = mean_random_forest.predict(mean_valid_data)
    mean_accuracy += [accuracy_score(valid_label, mean_valid_pred)]

list(zip(n_estimators, mean_accuracy))
mean_best_n_estimator = n_estimators[np.argmax(mean_accuracy)]
print(f"Best n_estimator for mean data is {mean_best_n_estimator}, it's valid accuracy is {max(mean_accuracy):.4f}")

# 3.1.2 비슷한 데이터들의 평균으로 채운 데이터
cluster_accuracy = []
for n_estimator in n_estimators:
    cluster_random_forest = RandomForestClassifier(n_estimators=n_estimator)
    cluster_random_forest.fit(cluster_train_data, train_label)
    cluster_valid_pred = cluster_random_forest.predict(cluster_valid_data)
    cluster_accuracy += [accuracy_score(valid_label, cluster_valid_pred)]

list(zip(n_estimators, mean_accuracy))
cluster_best_n_estimator = n_estimators[np.argmax(cluster_accuracy)]
print(f"Best n_estimator for cluster data is {cluster_best_n_estimator}, it's valid accuracy is {max(cluster_accuracy):.4f}")

# 3.1.3 Best Parameter
mean_random_forest = RandomForestClassifier(n_estimators=mean_best_n_estimator)
cluster_random_forest = RandomForestClassifier(n_estimators=cluster_best_n_estimator)

# 3.2 학습
mean_random_forest.fit(mean_train_data, train_label)
cluster_random_forest.fit(cluster_train_data, train_label)

# 3.3 예측
mean_test_pred = mean_random_forest.predict(mean_test_data)
cluster_test_pred = cluster_random_forest.predict(cluster_test_data)

# 3.4 평가
mean_test_accuracy = accuracy_score(test_label, mean_test_pred)
cluster_test_accuracy = accuracy_score(test_label, cluster_test_pred)

print(f"Test Accuracy for mean data is {mean_test_accuracy:.4f}")
print(f"Test Accuracy for cluster data is {cluster_test_accuracy:.4f}")

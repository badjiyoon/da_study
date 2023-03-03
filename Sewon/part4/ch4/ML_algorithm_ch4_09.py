#CH04_09. 손글씨 분류 (Python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
"""
손글씨 데이터는 0~9 까지의 숫자를 손으로 쓴 데이터입니다.
데이터는 sklearn.datasets의 load_digits 를 이용해 받을 수 있습니다.
"""

from sklearn.datasets import load_digits

digits = load_digits()

data, target = digits["data"], digits["target"]

#1.2 Data EDA
#데이터는 각 픽셀의 값을 나타냅니다.
data[0], target[0]

#데이터의 크기를 확인하면 64인데 이는 8*8 이미지를 flatten 시켰기 때문입니다.
data[0].shape

#실제로 0부터 9까지의 데이터를 시각화하면 다음과 같이 나타납니다.
samples = data[:10].reshape(10, 8, 8)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    axes[idx//5, idx%5].imshow(sample, cmap="gray")
plt.show()

#1.3 Data split

from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_target)}, {len(train_target)/len(data):.2f}")
print(f"test_data size: {len(test_target)}, {len(test_target)/len(data):.2f}")

#2. Random Forest

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()

#2.1 학습

random_forest.fit(train_data, train_target)

#2.2 Feature Importance

random_forest.feature_importances_
feature_importance = pd.Series(random_forest.feature_importances_)
feature_importance.head(10)

feature_importance = feature_importance.sort_values(ascending=False)
feature_importance.head(10)
feature_importance.head(10).plot(kind="barh")
plt.show()

image = random_forest.feature_importances_.reshape(8, 8)

plt.imshow(image, cmap=plt.cm.hot, interpolation="nearest")
cbar = plt.colorbar(ticks=[random_forest.feature_importances_.min(), random_forest.feature_importances_.max()])
cbar.ax.set_yticklabels(['Not Important', 'Very Important'])
plt.axis("off")
plt.show() #픽셀의 색이 밝을수록 중요한 변수


#2.3 예측
train_pred = random_forest.predict(train_data)
test_pred = random_forest.predict(test_data)

#실제 데이터를 한 번 그려보겠습니다.
plt.imshow(train_data[4].reshape(8, 8), cmap="gray")
plt.show() #9
train_pred[4]

#2.4 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

print(f"train accuracy is {train_acc:.4f}")
print(f"test accuracy is {test_acc:.4f}")

#3. Best Hyper Parameter
"""
RandomForestClassifier에서 주로 탐색하는 argument들은 다음과 같습니다.

n_estimators: 몇 개의 나무를 생성할 것 인지 정합니다.
criterion: 어떤 정보 이득을 기준으로 데이터를 나눌지 정합니다. (gini, entropy)
max_depth: 나무의 최대 깊이를 정합니다.
min_samples_split: 노드가 나눠질 수 있는 최소 데이터 개수를 정합니다.

탐색해야할 argument들이 많을 때 일일이 지정을 하거나 for loop을 작성하기 힘들어집니다.
이 때 사용할 수 있는 것이 sklearn.model_selection의 GridSearchCV 함수입니다.
"""

#GridSearchCV: 머신러닝에서 모델의 성능 향상을 위해 쓰이는 기법
#사용자가 직접 모델의 하이퍼 파라미터의 값을 리스트로 입력하면 값에 대한 경우의 수마다 예측 성능을 측정
from sklearn.model_selection import GridSearchCV

#3.1 탐색 범위 선정
#탐색할 값들의 argument와 범위를 정합니다.
params = {
    "n_estimators": [i for i in range(100, 1000, 200)],
    "max_depth": [i for i in range(10, 50, 10)],
}
params
#max_depth: 10, 20, 30, 40 / n_estimators: 100, 200, 300, 400, 500


#탐색에 사용할 모델을 생성합니다.
random_forest = RandomForestClassifier()

#3.2 탐색
#탐색을 시작합니다. cv는 k-fold의 k값
grid = GridSearchCV(estimator=random_forest, param_grid=params, cv=3)
# 4*5=20개의 경우의 수* CV 3 = 60개 모델 확인
grid = grid.fit(train_data, train_target)
grid

#3.3 결과
print(f"Best score of paramter search is: {grid.best_score_:.4f}")
grid.best_params_

print("Best parameter of best score is")
print(f"\t max_depth: {grid.best_params_['max_depth']}")
print(f"\t n_estimators: {grid.best_params_['n_estimators']}")

best_rf = grid.best_estimator_
best_rf

#3.4 예측
train_pred = best_rf.predict(train_data)
test_pred = best_rf.predict(test_data)

#3.5 평가
best_train_acc = accuracy_score(train_target, train_pred)
best_test_acc = accuracy_score(test_target, test_pred)

print(f"Best parameter train accuracy is {best_train_acc:.4f}")
print(f"Best parameter test accuracy is {best_test_acc:.4f}")

print(f"train accuracy is {train_acc:.4f}")
print(f"test accuracy is {test_acc:.4f}")

#4. Feature Importance
best_feature_importance = pd.Series(best_rf.feature_importances_)
best_feature_importance = best_feature_importance.sort_values(ascending=False)
best_feature_importance.head(10)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
feature_importance.head(10).plot(kind="barh", ax=axes[0], title="Random Forest Feature Importance")
best_feature_importance.head(10).plot(kind="barh", ax=axes[1], title="Best Parameter Feature Importance")
plt.show()
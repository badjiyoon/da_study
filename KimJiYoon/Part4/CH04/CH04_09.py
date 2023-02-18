import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

np.random.seed(2021)

# 1. Data
# 손 글씨 데이터는 0-9까지의 숫자를 손으로 쓴 데이터
digits = load_digits()

data, target = digits['data'], digits['target']

# 1.2 Data EDA
print('data[0] : ', data[0])
print('target[0] : ', target[0])
# 데이터의 크기를 확인하면 64인데 8*8 이미지를 flatten 시켰음
print('data[0].shape', data[0].shape)
# 실제로 0부터 9까지의 데이터를 시각화하면 다음과 같이 나타남
samples = data[:10].reshape(10, 8, 8)
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(20, 10))
for idx, sample in enumerate(samples):
    axes[idx // 5, idx % 5].imshow(sample, cmap='gray')
plt.show()

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)

print(f'train_data size : {len(train_target)}, {len(train_target) / len(data): .2f}')
print(f'test_data size : {len(test_target)}, {len(test_target) / len(data): .2f}')

# 2. Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(train_data, train_target)

# 2.2 Feature Importance
print(f'random_forest.feature_importances_ : {random_forest.feature_importances_}')
feature_importance = pd.Series(random_forest.feature_importances_)
print(f'feature_importance head : {feature_importance.head(10)}')
feature_importance = feature_importance.sort_values(ascending=False)
print(f'feature_importance head : {feature_importance.head(10)}')
feature_importance.head(10).plot(kind='barh')
plt.show()

image = random_forest.feature_importances_.reshape(8, 8)

plt.imshow(image, cmap=plt.cm.hot, interpolation='nearest')
cbar = plt.colorbar(ticks=[random_forest.feature_importances_.min(), random_forest.feature_importances_.max()])
cbar.ax.set_yticklabels(['Hot Important', 'Very Important'])
plt.axis('off')
plt.show()

# 2.3 예측
train_pred = random_forest.predict(train_data)
test_pred = random_forest.predict(test_data)
# 실제 데이터 보기
plt.imshow(train_data[4].reshape(8, 8), cmap="gray")
plt.show()
print(f'train_pred[4] : {train_pred[4]}')
# 2.4 평가
train_acc = accuracy_score(train_target, train_pred)
test_acc = accuracy_score(test_target, test_pred)

# 3. Best Hyper Prameter
# RandomForestClassifier에서 주로 탐색하는 Argument
# n_estimators : 몇개의 나무를 생성할 것 인지 정합니다.
# criterion : 어떤 정보 이득을 기준으로 데이터를 나눌지 정합니다. 'gini', 'entropy'
# max_depth : 나무의 최대 길이를 정합니다.
# min_samples_split : 노드가 나눠질 수 있는 최소 데이터 개수를 정합니다.
# 탐색해야할 Argument들이 많을 때 일일이 지정하거나 for loop를 작성하기 힘들어진다.
# 이 때 사용할 수 있는 것이 skearn.model_selection의 GridSearchCV 함수이다.

# 3.1 탐색 범위 선정
# 탐색할 값들의 Argument와 범위를 정합니다.
params = {
    "n_estimators": [i for i in range(100, 1000, 200)],
    "max_depth": [i for i in range(10, 50, 10)],
}

print(f'param : {params}')
random_forest = RandomForestClassifier()

# 3.2 탐색
# 탐색 시작 cv는 k-fold의 K값
grid = GridSearchCV(estimator=random_forest, param_grid=params, cv=3)
grid = grid.fit(train_data, train_target)

# 3.3 결과
print(f"Best score of paramter search is: {grid.best_score_:.4f}")
print("Best parameter of best score is")
print(f"\t max_depth: {grid.best_params_['max_depth']}")
print(f"\t n_estimators: {grid.best_params_['n_estimators']}")

best_rf = grid.best_estimator_

# 3.4 에측
train_pred = best_rf.predict(train_data)
test_pred = best_rf.predict(test_data)

# 3.5 평가
best_train_acc = accuracy_score(train_target, train_pred)
best_test_acc = accuracy_score(test_target, test_pred)
print(f'Best prameter train accuracy is {best_train_acc: .4f}')
print(f'Best prameter test accuracy is {best_test_acc: .4f}')

# 4. Feature Importance
best_feature_importance = pd.Series(best_rf.feature_importances_)
best_feature_importance = best_feature_importance.sort_values(ascending=False)
print(f'best_feature_importance.head(10) : {best_feature_importance.head(10)}')
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
feature_importance.head(10).plot(kind='barh', ax=axes[0], title='Random Forest Feature Importance')
best_feature_importance.head(10).plot(kind='barh', ax=axes[0], title='Random Forest Feature Importance')

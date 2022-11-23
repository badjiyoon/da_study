# ㄷㅏㅇ뇨병 진행도와 관련된 Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# Data
from sklearn.datasets import load_diabetes

# bmi Body mass index
# bp Average blood pressure
# 혈청에 대한  6가지 지표들 s1, s2, s3, s4 , s5, s6
diabetes = load_diabetes()
print(diabetes['feature_names'])

# 데이터와 정답 확인
data, target = diabetes['data'], diabetes['target']
print(data[0])

# Data EDA
df = pd.DataFrame(data, columns=diabetes['feature_names'])
print(df.describe())

# Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)
print(len(data), len(train_data), len(test_data))
print('train ratio : {:.2f}'.format(len(train_data) / len(data)))
print('test ratio : {:.2f}'.format(len(test_data) / len(data)))

# 학습
from sklearn.linear_model import LinearRegression

multi_regressor = LinearRegression()
multi_regressor.fit(train_data, train_target)

# 회귀식 확인
print(multi_regressor.intercept_)
print(multi_regressor.coef_)

# 3 예측
multi_train_pred = multi_regressor.predict(train_data)
multi_test_pred = multi_regressor.predict(test_data)

# 평가
from sklearn.metrics import mean_squared_error
multi_train_mse = mean_squared_error(multi_train_pred, train_target)
multi_test_mse = mean_squared_error(multi_test_pred, test_target)

# Ridge Regression
from sklearn.linear_model import Ridge
ridge_regressor = Ridge()
ridge_regressor.fit(train_data, train_target)

# 회귀식확인
print(ridge_regressor.intercept_)
print(ridge_regressor.coef_)

# 예측
ridge_train_pred = ridge_regressor.predict(train_data)
ridge_test_pred = ridge_regressor.predict(test_data)

# 평가
ridge_train_mse = mean_squared_error(ridge_train_pred, train_target)
ridge_test_mse = mean_squared_error(ridge_test_pred, test_target)

print(f'Ridge Regression Train MSE is {ridge_train_mse: .4f}')
print(f'Ridge Regression Test MSE is {ridge_test_mse: .4f}')

# LASSO Regression
from sklearn.linear_model import Lasso

lasso_regressor = Lasso()
lasso_regressor.fit(train_data, train_target)

# 회귀식 확인
print(lasso_regressor.intercept_)
print(lasso_regressor.coef_)

np.array(diabetes['feature_names'])[lasso_regressor.coef_ != 0]

# 예측
lasso_train_pred = lasso_regressor.predict(train_data)
lasso_test_pred = lasso_regressor.predict(test_data)

# 평가
lasso_train_mse = mean_squared_error(lasso_train_pred, train_target)
lasso_test_mse = mean_squared_error(lasso_test_pred, test_target)

print(f'Lasso Regression Train MSE is {lasso_train_mse: .4f}')
print(f'Lasso Regression Test MSE is {lasso_test_mse: .4f}')

# 마무리
print(f'Multi Regression Test MSE is {multi_test_mse: .4f}')
print(f'Ridge Regression Test MSE is {ridge_test_mse: .4f}')
print(f'Lasso Regression Test MSE is {lasso_test_mse: .4f}')

# 에측값과 실제값의 관계 Plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
preds = [
    ('Multi regression', multi_test_pred),
    ('Ridge regression', ridge_test_pred),
    ('LASSO regression', lasso_test_pred)
]

for idx, (name, test_pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(test_pred, test_target)
    ax.plot(np.linspace(0, 330, 100), np.linspace(0, 330, 100), color='red')
    ax.set_xlabel('Predict')
    ax.set_ylabel('Real')
    ax.set_title(name)

plt.show()

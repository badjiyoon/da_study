# 당뇨병 진행도와 관련된 Linear Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

np.random.seed(2021)

# bmi Body mass index
# bp Average blood pressure
# 혈청에 대한  6가지 지표들 s1, s2, s3, s4 , s5, s6
# age : age in years
# sex : 성별
# bmi : body mass index BMI / 지수
# bp : average blood pressure / 혈압 평균
# s1 tc, total serum cholesterol / 총 혈청 콜레스테롤
# s2 ldl, low-density lipoproteins / 저밀도 단백질
# s3 hdl, high-density lipoproteins / 고밀도 단백질
# s4 tch, total cholesterol / HDL 전체 콜레스테롤
# s5 ltg, possibly log of serum triglycerides level / 아마도 혈청 트리글리세리드 수준의 로그
# s6 glu, blood sugar level / 혈당 수치

diabetes = load_diabetes()
print('diabetes["feature_names"] : ', diabetes['feature_names'])
print('diabetes["feature_names"] len : ', len(diabetes['feature_names']))

# 데이터와 정답 확인
data, target = diabetes['data'], diabetes['target']
print('0 번째 데이터 : ', data[0])
print('0 번째 데이터 정답 : ', target[0])

# Data EDA
df = pd.DataFrame(data, columns=diabetes['feature_names'])
print('데이터 null 값 체크 : ', df.isnull())
print('데이터프레임 통계량 : ', df.describe())

# Data Split 데이터 분할
train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)
print(len(data), len(train_data), len(test_data))
print('train 비율  : {:.2f}'.format(len(train_data) / len(data)))
print('test 비율 : {:.2f}'.format(len(test_data) / len(data)))

# 학습
multi_regressor = LinearRegression()
multi_regressor.fit(train_data, train_target)

# 회귀식 확인
print(multi_regressor.intercept_)
print(multi_regressor.coef_)

# 예측
multi_train_pred = multi_regressor.predict(train_data)
multi_test_pred = multi_regressor.predict(test_data)

# 평가 지표 확인 MSE (평균 제곱 오차)
# 평균제곱오차(Mean Squared Error, MSE)는 이름에서 알 수 있듯이 오차(error)를 제곱한 값의 평균입니다.
# 오차란 알고리즘이 예측한 값과 실제 정답과의 차이를 의미합니다. 즉, 알고리즘이 정답을 잘 맞출수록 MSE 값은 작겠죠.
# 즉, MSE 값은 작을수록 알고리즘의 성능이 좋다고 볼 수 있습니다. 수식을 살펴보겠습니다.
# https://heytech.tistory.com/362
multi_train_mse = mean_squared_error(multi_train_pred, train_target)
multi_test_mse = mean_squared_error(multi_test_pred, test_target)

# Ridge Regression
# 혼란스러운 것을 뭔가 제약을 주어서 정돈한다라는 의미로 regularization(정규화)
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

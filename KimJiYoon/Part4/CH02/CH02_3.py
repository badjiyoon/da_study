import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 단순회기
# Simple Date
X = np.array([1, 2, 3, 4])
y = np.array([2, 1, 4, 3])

plt.scatter(X, y)
plt.show()

# Data 변환
# scikit-learn에서 모델 학습을 위한 데이터는 (n,c) 형태로 되어 있어야합니다.
# 1. n은 데이터의 개수를 의미
# 2. c는 feature의 개수를 의미
# 현재 데이터는 4개의 데이터와 1개의 feature로 이루어져 있습니다.
print(X)
print(X.shape)
data = X.reshape(-1, 1)
print(data.shape)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
# 학습하기
# scikit-learn 패키지의 LinearRegression을 이용해 선형 회귀 모델을 생성
# model을 학습 fit함수를 이용해 가능

model.fit(X=data, y=y)
# model.fit(data, y)

# 모델의 식 확인
# bias : 편향을 먼저 확인
print(model.intercept_)

# 다음은 회귀계수 확인
print(model.coef_)

# 위의 두 결과로 다음과 같은 회귀선을 얻을 수 있습니다.
# y = 1.0000000000000004 + 0.6 * x

# 예측하기
# 모델의 예측은 PREdict함수를 통해 할 수 있습니다.
pred = model.predict(data)
print(pred)

# 회귀선을 Plot으로 표현하기
plt.scatter(X, y)
plt.plot(X, pred, color='green')

# 멀티 회기
bias = 1
beta = np.array([2, 3, 4, 5]).reshape(4, 1)
noise = np.random.randn(100, 1)

X = np.random.randn(100, 4)
y = bias + X.dat(beta)
y_with_noise = y + noise
print(X[:10])
print(y[:10])

model = LinearRegression()
model.fit(X, y_with_noise)

# 회귀식 확인하기
print(model.intercept_)
print(model.coef_)

# 통계적 방법
bias_X = np.array([1] * len(X)).reshape(-1, 1)
stat_X = np.hstack([bias_X, X])
X_X_transpose = stat_X.transpose().dot(stat_X)
X_X_transpose_inverse = np.linalg.inv(X_X_transpose)

stat_beta = X_X_transpose_inverse.dot(stat_X.transpose()).dot(y_with_noise)
print(stat_beta)

# 3. Polynomial Regression
bias = 1
beta = np.array([2, 3]).reshape(2, 1)
noise = np.random.randn(100, 1)

X = np.random.randn(100, 1)
X_poly = np.hstack([X, X**2])
print(X_poly[:10])
Y = bias + X_poly.dot(beta)
y_with_noise = y + noise
plt.scatter(X, y_with_noise)
# 3-2 Polynomial Regression
model = LinearRegression()
model.fit(X_poly, y_with_noise)
# 회귀식 확인하기
print(model.intercept_)
print(model.coef_)
# 예측하기
pred = model.predict(X_poly)
# plot으로 확인하기
plt.scatter(X, pred)

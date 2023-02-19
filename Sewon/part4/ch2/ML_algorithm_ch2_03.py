# CH02_03. Linear Regression 실습하기(python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(2021) #seed: 랜덤 난수 생성

#1. Univariate Regression
#1.1 Sample data

X=np.array([1, 2, 3, 4])
y=np.array([2, 1, 4, 3])

plt.scatter(X, y)
plt.show()

#1.2 데이터 변환
'''
scikit-learn에서 모델 학습을 위한 데이터는 (n, c) 형태로 되어 있어야 함.
n: 데이터의 개수 (행), array
c: feature의 개수 (열)

*우리가 사용하는 데이터는 4개의 데이터와 1개의 feature로 이루어짐
'''
X
X.shape #그냥 array라서 (4, ) 로만 출력
data=X.reshape(-1, 1) #행은 개수대로 출력, 열은 1열로 보겠다
data
data.shape

'''
np.reshape 함수 기본 사용법
-기본적인 사용 방법은 배열 a에 대하여
a.reshape(변환 shape) 혹은 np.reshape(a, 변환 shape)형태로 사용
-axis 순서대로(가로 -> 세로 축 방향)값들을 변환되는 shape으로 재배정하는 원리
-재배정이 불가능한 shape인 경우 ValueError가 발생

*-1을 넣는 경우
-'-1'을 넣은 자리에는 가능한 shape을 자동 계산하여 반영해주는 방식으로,
예를 들어, 8개의 사이즈에서 reshape(2, -1)로 넣으면, (2, 4)로 자동 변환되는 방식입니다.
-단, 2개 이상의 axis 자리에 -1이 포함되는 것은 불가능
'''

#1.3 Linear Regression

import sklearn
from sklearn.linear_model import LinearRegression

model=LinearRegression() #LinearRegression을 model 변수에 선언(?)

#1.3.1 학습하기
'''
scikit-learn 패키지의 LinearRegression을 이용해 선형 회귀 모델 생성
model.fit(X=..., y=...)
'''
model.fit(X=data, y=y)
model.fit(data, y) #변수 빼고 써도 됨

#1.3.2 모델의 식 확인
'''
1) bias, 편향을 먼저 확인
-sklearn에서는 intercept_ 로 확인

2) 회귀계수 확인
-sklearn에서는 coef_ 로 확인
'''
model.intercept_ #편향
model.coef_ #회귀계수

#1.3.3 예측하기
'''
model.predict(X= ~ )
'''
pred=model.predict(data) 
pred

#1.4 회귀선을 Plot으로 표현하기
plt.scatter(X, y)
plt.plot(X, pred, color='pink') #plot(정의역, y값(=함수식), 옵션)
plt.show()


#2. Multivatiate Regression
#2.1 Sample data
'''
*randn(m, n)
-표준 정규 분포 추출 함수 
-m*n차원, 크기의 가우시안 분포의 난수 생성
-평균 0, 표준편차 1을 가지는 표준 정규 분포 내에서 임의 추출하는 함수
 (-2~2 사이의 값들에서 95% 가량 추출)
'''

bias=1
beta=np.array([2, 3, 4, 5]).reshape(4, 1)
noise=np.random.randn(100, 1) #randn: 기대값 0, 표준편차 1인 랜덤 난수 생성
noise #100행 1열 데이터 출력

X=np.random.randn(100, 4)
y=bias+X.dot(beta)
y_with_noise=y+noise #y값엔 보통 noise가 있다고 가정

X[:10] #슬라이싱: 처음부터 10-1번째까지 행, 열도 슬라이싱 할거면 [:] 이어서 씀
y[:10]

#2.2 Multivariate Regression
model=LinearRegression()
model.fit(X, y_with_noise)

#2.3 회귀식 확인
model.intercept_
model.coef_

#2.4 통계적 방법
'''
np.hstack() array 수평 방향 쌓기
np.vstack() array 수직 방향 쌓기
'''
X
len(X)
bias_X=np.array([1]*len(X)).reshape(-1, 1) #len(X)행 1열에 1만 들어있는 행렬을 만들고자 한 것
stat_X=np.hstack([bias_X, X]) 
stat_X
X_X_transpose=stat_X.transpose().dot(stat_X)
X_X_transpose_inverse=np.linalg.inv(X_X_transpose) 

stat_beta=X_X_transpose_inverse.dot(stat_X.transpose()).dot(y_with_noise)

stat_beta

#3. Polynomial Regression
#3.1 Sample data

bias=1
beta=np.array([2, 3]).reshape(2, 1)
noise=np.random.rand(100, 1)
beta

X=np.random.randn(100, 1)
X_poly=np.hstack([X, X**2]) #X**2=모든 X의 요소에 제곱을 한 것것
X_poly
bias

y=bias+X_poly.dot(beta)
y
y_with_noise=y+noise

plt.scatter(X, y_with_noise)
plt.show()

#3.2 Polynomial Regression
#3.2.1 학습하기
model=LinearRegression()
model.fit(X_poly, y_with_noise)

#3.2.2 회귀식 확인
model.intercept_ #y절편
model.coef_ #계수, 기울기

#3.2.3 예측하기
pred=model.predict(X_poly)

#3.3 예측값을 Plot으로 확인
plt.scatter(X, pred)
plt.show()
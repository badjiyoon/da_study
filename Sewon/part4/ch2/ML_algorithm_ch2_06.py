# CH02_06. 당뇨병 진행도 예측(python)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data load
from sklearn.datasets import load_diabetes

diabetes=load_diabetes()

type(diabetes)

'''
-당뇨병 데이터에서 사용되는 변수명은 feature_names 키 값으로 들어 있음
age: 나이
sex: 성별
bmi: Body Mass Index
bp: average blood pressure
혈정에 대한 6가지 지표: s1~s6
'''

diabetes["feature_names"]
data, target = diabetes["data"], diabetes["target"]
data[0]
target[0]

#1.2 Data EDA
df=pd.DataFrame(data, columns=diabetes["feature_names"])
df.describe()

#1.3 Data split
from sklearn.model_selection import train_test_split
'''
 train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
)

*arrays: 입력은 array로 이루어진 데이터을 받습니다.
test_size: test로 분할될 사이즈를 정합니다.
train_size: train으로 분할될 사이즈를 정합니다.
random_state: 다음에도 같은 값을 얻기 위해서 난수를 고정합니다
shuffle: 데이터를 섞을지 말지 결정합니다.
stratify: 데이터를 나눌 때 정답의 분포를 반영합니다.
'''

train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.3)
#train:test=7:3
len(data), len(train_data), len(test_data)

print("train ratio : {:.2f}".format(len(train_data)/len(data)))
print("test ratio : {:.2f}".format(len(test_data)/len(data)))

#2. Multivariate Regression
#2.1 학습
from sklearn.linear_model import LinearRegression

multi_regressor = LinearRegression()
multi_regressor.fit(train_data, train_target)

#2.2 회귀식 확인
multi_regressor.intercept_
multi_regressor.coef_

#2.3 예측
multi_train_pred = multi_regressor.predict(train_data)
multi_test_pred = multi_regressor.predict(test_data)

#2.4 평가
#mean_squared_error
from sklearn.metrics import mean_squared_error

multi_train_mse = mean_squared_error(multi_train_pred, train_target)
multi_test_mse = mean_squared_error(multi_test_pred, test_target)

print(f"Multi Regression Train MSE is {multi_train_mse:.4f}")
print(f"Multi Regression Test MSE is {multi_test_mse:.4f}")

#3. Ridge Regression
#3.1 학습
from sklearn.linear_model import Ridge

ridge_regressor = Ridge()
ridge_regressor.fit(train_data, train_target)

#3.2 회귀식 확인
ridge_regressor.intercept_
multi_regressor.coef_
ridge_regressor.coef_

#3.3 예측
ridge_train_pred = ridge_regressor.predict(train_data)
ridge_test_pred = ridge_regressor.predict(test_data)

#3.4 평가
ridge_train_mse = mean_squared_error(ridge_train_pred, train_target)
ridge_test_mse = mean_squared_error(ridge_test_pred, test_target)

print(f"Ridge Regression Train MSE is {ridge_train_mse:.4f}")
print(f"Ridge Regression Test MSE is {ridge_test_mse:.4f}")

#4. LASSO Regression
#4.1 학습
from sklearn.linear_model import Lasso

lasso_regressor = Lasso()
lasso_regressor.fit(train_data, train_target)

#4.2 회귀식 확인
lasso_regressor.intercept_
lasso_regressor.coef_

np.array(diabetes["feature_names"])[lasso_regressor.coef_ != 0]

#4.3 예측
lasso_train_pred = lasso_regressor.predict(train_data)
lasso_test_pred = lasso_regressor.predict(test_data)

#4.4 평가
lasso_train_mse = mean_squared_error(lasso_train_pred, train_target)
lasso_test_mse = mean_squared_error(lasso_test_pred, test_target)

print(f"LASSO Regression Train MSE is {lasso_train_mse:.4f}")
print(f"LASSO Regression Test MSE is {lasso_test_mse:.4f}")

#5. 마무리
#5.1 평가
print(f"Multi Regression Test MSE is {multi_test_mse:.4f}")
print(f"Ridge Regression Test MSE is {ridge_test_mse:.4f}")
print(f"LASSO Regression Test MSE is {lasso_test_mse:.4f}")

#5.2 예측값과 실제값의 관계 Plot
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
preds = [
    ("Multi regression", multi_test_pred),
    ("Ridge regression", ridge_test_pred),
    ("LASSO regression", lasso_test_pred),
]

for idx, (name, test_pred) in enumerate(preds):
    ax = axes[idx]
    ax.scatter(test_pred, test_target)
    ax.plot(np.linspace(0, 330, 100), np.linspace(0, 330, 100), color="red")
    ax.set_xlabel("Predict")
    ax.set_ylabel("Real")
    ax.set_title(name)
plt.show()
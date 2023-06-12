# 01. 데이터 소개 및 분석프로세스 수립 "강의자료 → Ch03. [실습1] 보험료 예측하기" 참고

# 라이브러리 로드
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns
import missingno

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score

# 02. 데이터 준비를 위한 EDA 및 전처리
# [EDA 체크리스트]
# 1. 어떤 질문을 풀거나 틀렸다고 증명하려고 하는가?
# 2. 중복된 항목은 있는가?
# 3. 어떤 종류의 데이터가 있으며 다른 데이터 타입들을 어떻게 다루려고 하는가?
# 4. 데이터에서 누락된 것이 있는지, 있다면 그것들을 어떻게 처리하려는가?
# 5. 이상치는 어디에 있는가? 관심을 가져야 할 데이터인가?
# 6. 변수 간 상관성이 있는가?

# 0. 데이터 불러오기
data = pd.read_csv("../../../comFiles/ch3_premium.csv")

# 1. 어떤 질문을 풀거나 틀렸다고 증명하려고 하는가?
#### 보험사 고객 정보를 통해 보험료 예측 모델 생성 ? ####
# 간략한 데이터 살피기
print(data.shape)
# 데이터 15개 행 데이터 확인
print(data.head(15))
# 고객ID처럼 명백하게 보험료와 관계없는 것은 없는가?
# 컬럼 중 의미가 이해가지 않는 것은 없는가?
# 약어나 전문 용어로 되어 있는 것은 없는가?
# 2. 중복된 항목은 있는가?
# 중복된 항목 수 알아보기
print(data.duplicated())  # 각 행마다의 중복값 True False 값을 리턴해줌
print("중복된 항목 수 : ", len(data[data.duplicated()]))
# 중복된 항목 확인 (keep 옵션 -> 유지한다) -> 많은 케이스의 경우 정렬
print(data[data.duplicated(keep=False)].sort_values(by=list(data.columns)).head())
# 중복된 항목 제거
# data = data.drop_duplicates(inplace=True, keep='first', ignore_index=True)
# 3. 어떤 종류의 데이터가 있으며 다른 데이터 타입들을 어떻게 다루려고 하는가?
# 총 컬럼 수와 컬럼별 데이터 타입 확인
print(data.info())
# 데이터 타입별 컬럼 수 확인하기
print(data.dtypes)
# 인덱스를 날림
dtype_data = data.dtypes.reset_index()
# 컬럼을 새로 생성함
dtype_data.columns = ["Count", "Columns Type"]
dtype_data = dtype_data.groupby("Columns Type").aggregate("count").reset_index()
print(dtype_data)

# 숫자형 데이터 중 명백하게 포함할 의미가 없는 것은 없는가?
# 범주형 변수는 있는가?
# 범주형 변수별 개수 시각화
# object 각각의 컬럼을 활용
for col in data.select_dtypes(include=["object", "category"]).columns:
    fig = seaborn.catplot(x=col, kind="count", data=data, hue=None)
    fig.set_xticklabels(rotation=90)
    plt.show()

# 데이터 컬럼별 유일한 개수 확인하기
# nunique()는 데이터에 고유값들의 수를 출력해주는 함수
print(data.select_dtypes(include=["object", "category"]).nunique())

# 항목이 2개인 성별과, 흡연 여부는 LabelEncoder 를, 지역은 OneHotEncoder 를 사용 하기로 한다.
# sklearn 의 LabelEncoder, OneHotEncoder 사용

# LabelEncoder : 각각의 범주를 서로 다른 정수로 맵핑
# 성별, 흡연 여부 컬럼은 Label Encoding 을 위해 ndarray 로 변환 하여 준다
sex = data.iloc[:, 1:2].values
smoker = data.iloc[:, 4:5].values

### 성별 ###
# 1. LabelEncoder() 를 선언
le = LabelEncoder()
# 2. 성별을 LabelEncoder 의 fit_transform 에 넣어 준다
sex[:, 0] = le.fit_transform(sex[:, 0])
print(type(sex[:, 0]))
sex = pd.DataFrame(sex[:, 0])
sex.columns = ["sex"]
# 3. dict 형으로 변환
# zip() 여러 개의 순회 가능한(iterable) 객체를 인자로 받고,
# 각 객체가 담고 있는 원소를 튜플의 형태로 차례로 접근할 수 있는 반복자(iterator)를 반환
lex_sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("성별에 대한 Label Encoder 결과 : ", lex_sex_mapping)
print(sex[:10])
### 흡연 여부 ###
# 1. LabelEncoder() 를 선언
le = LabelEncoder()
# 2. 흡연 여부를 LabelEncoder 의 fit_transform 에 넣어 준다
smoker[:, 0] = le.fit_transform(smoker[:, 0])
print(type(smoker[:, 0]))
smoker = pd.DataFrame(smoker[:, 0])
smoker.columns = ["smoker"]
# 3. dict 형으로 변환
lex_smoker_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("흡연에 대한 Label Encoder 결과 : ", lex_smoker_mapping)
print(smoker[:10])

# OneHot Encoder : 각각의 범주를 0과 1로 맵핑
# 지역 컬럼은 Label Encoding 을 위해 ndarray 로 변환
region = data.iloc[:, 5:6].values
### 지역 ###
# 1. OneHotEncoder() 를 선언해주고
ohe = OneHotEncoder()
# 2. 지역을 OneHotEncoder 의 fit_transform 에 넣어준다
print(ohe.fit_transform(region).toarray())
region = ohe.fit_transform(region).toarray()
region = pd.DataFrame(region)
region.columns = ["northeast", "northwest", "southeast", "southwest"]
print("지역에 대한 One Hot Encoder 결과 : \n", region[:10])

# 4. 데이터에서 누락된 것이 있는지, 있다면 그것들을 어떻게 처리하려는가?
# NULL 값이 포함된 컬럼 찾기 -> 각 컬럼의 평균값으로 채우기 (Imputation 또는 보간법)
# 각 컬럼들에 몇 개의 NULL 값이 포함되어 있는지 확인
count_nan = data.isnull().sum()
print(count_nan[count_nan > 0])
# missingno 패키지를 통해 시각화 확인
missingno.matrix(data, figsize=(30, 10))
plt.show()
# seaborn 패키지 heatmap 을 통해 시각화 확인
sns.heatmap(data.isnull(), cbar=False, yticklabels=False, cmap="viridis")
plt.show()
# NULL 값을 해당 컬럼의 평균값으로 대체하기
data["bmi"].fillna(data["bmi"].mean(), inplace=True)
print(data.head(15))
# 확인
count_nan = data.isnull().sum()
print(count_nan[count_nan > 0])
# missingno 패키지를 통해 시각화 재확인
missingno.matrix(data, figsize=(30, 10))
plt.show()
# * 결측값 처리 참고 사이트 : https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4
### 5. 이상치는 어디에 있는가? 관심을 가져야 할 데이터인가?
# 숫자형 데이터별 요약 통계값 확인 (간단한 통계값 T의 경우 trans)
print(data.describe().T)
# 데이터 컬럼별 요약 통계값 보기 (기본적인 그래프 -> )
data.age.plot.hist()
plt.show()
# 데이터 개별 컬럼 히스토그램으로 확인하기
import scipy

scipy.__version__
# > 숫자형 데이터 Skewness 확인 (외도)
# 데이터 컬럼 타입이 np.number 인 것만 가져오기 (숫자형)
numeric_data = data.select_dtypes(include=np.number)
# 데이터 컬럼 타입이 np.number 인 컬럼 이름들 가져오기
l = numeric_data.columns.values
number_of_columns = 4
number_of_rows = len(l) - 1 / number_of_columns
print(number_of_rows)
# 컬럼별 히스토그램 그리기 (kde => 경향)
for i in range(0, len(l)):
    sns.displot(numeric_data[l[i]], kde=True)
    plt.show()
# > 숫자형 데이터 Box Plot 시각화
# 데이터 컬럼 타입이 np.number 인 컬럼들 가져오기
columns = data.select_dtypes(include=np.number).columns
figure = plt.figure(figsize=(20, 10))
figure.add_subplot(1, len(columns), 1)
for index, col in enumerate(columns):
    if index > 0:
        figure.add_subplot(1, len(columns), index + 1)
    sns.boxplot(y=col, data=data, boxprops={"facecolor": "None"})
figure.tight_layout()  # 자동으로 명시된 여백에 관련된 서브플롯 파라미터를 조정
plt.show()

plt.figure(figsize=(20, 20))
for i in range(0, len(l)):
    print('number_of_rows ', number_of_rows)
    # 안됨
    # plt.subplot(number_of_rows + 1, number_of_columns, i + 1)
    plt.subplot(1, number_of_columns, i + 1)
    sns.set_style('whitegrid')
    sns.boxplot(numeric_data[l[i]], color='green', orient='v')
    plt.tight_layout()
plt.show()

# > 범주형 데이터별 Violin Plot 시각화
if len(data.select_dtypes(include=['object', 'category']).columns) > 0:
    for col_num in data.select_dtypes(include=np.number).columns:
        for col in data.select_dtypes(include=['object', 'category']).columns:
            fig = sns.catplot(x=col, y=col_num, kind='violin', data=data, height=5, aspect=2)
            fig.set_xticklabels(rotation=90)
            plt.show()

### 6. 변수 간 상관성이 있는가?
# > 숫자형 데이터 간 Pairwise 결합 분포 시각화
# Seaborn Heatmap 을 사용한 Correlation 시각화
# Seaborn Heatmap 을 사용한 Correlation 시각화
plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(numeric_only=True), cmap='Blues', annot=False)
plt.show()
# 판다스 업데이트 후 정수혐만 쓸것인지에 대한 옵션 처리
k = 4
cols = data.corr(numeric_only=True).nlargest(k, "charges")["charges"].index
cm = data[cols].corr(numeric_only=True)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, cmap="viridis", annot=True)
plt.show()
# 숫자 변수형 컬럼들 간 Pairplot 그리기 -> 선형인지 확인하는 것
sns.pairplot(data.select_dtypes(include=np.number))
plt.show()

# > 범주형 데이터를 기준으로 추가한 시각화
# https://seaborn.pydata.org/examples/index.html
hue = 'smoker'
sns.pairplot(data.select_dtypes(include=np.number).join(data[[hue]]), hue=hue)
plt.show()

## 03. 다양한 Regression 을 활용한 보험료 예측
# https://scikit-learn.org/stable/

### Training, Test 데이터 나누기

# 숫자형 데이터들만 copy() 를 사용하여 복사
X_num = data[["age", "bmi", "children"]].copy()
# 변환했던 범주형 데이터들과 concat 을 사용하여 합치기
X_final = pd.concat([X_num, region, sex, smoker], axis=1)
# 보험료 컬럼(charges)을 y 값으로 설정
y_final = data[["charges"]].copy()
# train_test_split 을 사용하여 Training, Test 나누기 (Training:Test=2:1)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)
X_train[0:10]
X_test[0:10]

# Feature Scaling
# *   다차원의 값들을 비교 분석하기 쉽게 만든다.
# *   변수들 간의 단위 차이가 있을 경우 필요하다.
# *   Overflow, Underflow 를 방지해준다.
## MinMaxScaler 를 사용하는 경우 : 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다

# n_scaler = MinMaxScaler()
# X_train = n_scaler.fit_transform(X_train.astype(np.float))
# X_test= n_scaler.transform(X_test.astype(np.float))

## StandardScaler 를 사용하는 경우 : 이상치가 있는 경우에는 균형 잡힌 결과를 보장하기 힘들다
s_scaler = StandardScaler()
X_train = s_scaler.fit_transform(X_train.astype(np.float64))
X_test = s_scaler.transform(X_test.astype(np.float64))

## 그 외 - RobustScaler 를 사용하는 경우 : 이상치의 영향을 최소화한 기법. 중앙값과 IQR 을 사용하기 때문에 표준화 후 동일한 값을 더 넓게 분포시키게 된다.

### Regression 절차 요약
# *   ****Regression()
# *   fit()
# *   predict()
# *   score()

### Linear Regression 적용
# fit model
lr = LinearRegression().fit(X_train, y_train)

# predict
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

# Score 확인
print("lr.coef_: {}".format(lr.coef_))
print("lr.intercept_: {}".format(lr.intercept_))
print('lr train score %.3f, lr test score: %.3f' % (lr.score(X_train, y_train), lr.score(X_test, y_test)))

### Polynomial Regression 적용
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_final)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_final, test_size=0.3, random_state=0)

# Standar Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float64))
X_test = sc.transform(X_test.astype(np.float64))
# fit model

poly_lr = LinearRegression().fit(X_train, y_train)

# predict
y_train_pred = poly_lr.predict(X_train)
y_test_pred = poly_lr.predict(X_test)

# Score 확인
print('poly train score %.3f, poly test score: %.3f' % (poly_lr.score(X_train, y_train), poly_lr.score(X_test, y_test)))

### Support Vector Regression 적용
svr = SVR(kernel="linear", C=300)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)

# Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float64))
X_test = sc.transform(X_test.astype(np.float64))

# fit model
svr = svr.fit(X_train, y_train.values.ravel())
y_train_pred = svr.predict(X_train)
y_test_pred = svr.predict(X_test)

# Score 확인
print('svr train score %.3f, svr test score: %.3f' % (svr.score(X_train, y_train), svr.score(X_test, y_test)))

### RandomForest Regression 적용
forest = RandomForestRegressor(n_estimators=100,
                               criterion='squared_error',
                               random_state=1,
                               n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)

# Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float64))
X_test = sc.transform(X_test.astype(np.float64))

# fit model
forest.fit(X_train, y_train.values.ravel())
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)
# Score 확인

print('forest train score %.3f, forest test score: %.3f' % (
    forest.score(X_train, y_train),
    forest.score(X_test, y_test)))

### Decision Tree Regression 적용
dt = DecisionTreeRegressor(random_state=0)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=0)

# Standard Scaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train.astype(np.float64))
X_test = sc.transform(X_test.astype(np.float64))

# fit model
dt = dt.fit(X_train, y_train.values.ravel())
y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)

# Score 확인
print('dt train score %.3f, dt test score: %.3f' % (
    dt.score(X_train, y_train),
    dt.score(X_test, y_test)))

### 다양한 모델 성능 종합 비교
# 앞에서 만든 regressor 변수들과 라벨을 묶어서 하나의 리스트로 모으기
regressors = [(lr, 'Linear Regression'),
              (poly_lr, 'Polynomial Regression'),
              (svr, 'SupportVector Regression'),
              (forest, 'RandomForest Regression'),
              (dt, 'DecisionTree')]

# 각 regressor 변수들과 라벨 묶음을 차례로 fit -> predict -> score 로 처리해서 보여주기
for reg, label in regressors:
    print(80 * '_', '\n')
    reg = reg.fit(X_train, y_train.values.ravel())
    y_train_pred = reg.predict(X_train)
    y_test_pred = reg.predict(X_test)
    print(f'{label} train score %.3f, {label} test score: %.3f' % (
        reg.score(X_train, y_train),
        reg.score(X_test, y_test)))

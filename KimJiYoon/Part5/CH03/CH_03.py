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
print(data.duplicated())# 각 행마다의 중복값 True False 값을 리턴해줌
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

# 숫자형 데이터 중 명백하게 포함할 의가 없는 것은 없는가?
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

### 성별 ###
# 1. LabelEncoder() 를 선언
# 2. 성별을 LabelEncoder 의 fit_transform 에 넣어 준다
# 3. dict 형으로 변환

### 흡연 여부 ###
# 1. LabelEncoder() 를 선언
# 2. 흡연 여부를 LabelEncoder 의 fit_transform 에 넣어 준다
# 3. dict 형으로 변환

# OneHot Encoder : 각각의 범주를 0과 1로 맵핑
# 지역 컬럼은 Label Encoding 을 위해 ndarray 로 변환

### 지역 ###
# 1. OneHotEncoder() 를 선언해주고
# 2. 지역을 OneHotEncoder 의 fit_transform 에 넣어준다


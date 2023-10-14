# -*- coding: utf-8 -*-
"""[전체]Ch16. [실습14] 시계열 분석을 이용한 Drugstore 매출 예측.ipynb
> https://www.kaggle.com/c/rossmann-store-sales
* [store.csv]
* [test.csv]
* [train.csv]
#Part4. [실습14] 시계열 분석을 이용한 Drugstore 매출 예측
"""

from matplotlib import pyplot as plt

plt.rc('font', family='AppleGothic')

"""## 01. 데이터 소개 및 분석프로세스 수립
## 02. 데이터 준비를 위한 EDA 및 전처리
"""

# Commented out IPython magic to ensure Python compatibility.
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.distributions.empirical_distribution import ECDF

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from prophet import Prophet

"""### 0. 데이터 불러오기"""

train = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/CH16/train.csv",
                    parse_dates=True, low_memory=False, index_col='Date')

store = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/CH16/store.csv",
                    low_memory=False)

train.head()
train.index

print("데이터 형태 : ", train.shape)
train.head(5)

"""### 새로운 컬럼 생성"""
train['Year'] = train.index.year
train['Month'] = train.index.month
train['Day'] = train.index.day
train['SalePerCustomer'] = train['Sales'] / train['Customers']
train['SalePerCustomer'].describe()

"""### 누적 분포 그래프"""
sns.set(style="ticks")
c = '#386B7F'
plt.figure(figsize=(12, 6))

plt.subplot(311)
cdf = ECDF(train['Sales'])
plt.plot(cdf.x, cdf.y, label="statmodels", color=c);
plt.xlabel('Sales')
plt.ylabel('ECDF')

plt.subplot(312)
cdf = ECDF(train['Customers'])
plt.plot(cdf.x, cdf.y, label="statmodels", color=c)
plt.xlabel('Customers')

plt.subplot(313)
cdf = ECDF(train['SalePerCustomer'])
plt.plot(cdf.x, cdf.y, label="statmodels", color=c)
plt.xlabel('Sale per Customer')

plt.show()

"""### 필터링"""

train[(train.Open == 0) & (train.Sales == 0)].head()
train.columns

zero_sales = train[(train.Open != 0) & (train.Sales == 0)]
print("데이터 형태 : ", zero_sales.shape)
zero_sales.head(5)

# 폐업을 한 상점 제외
train = train[(train["Open"] != 0) & (train['Sales'] != 0)]

print("데이터 형태 : ", train.shape)
train.columns
store.head()

"""### 결측값"""

store.isnull().sum()

store[pd.isnull(store.CompetitionDistance)]

"""> 결측값 처리"""

sns.displot(store['CompetitionDistance'], kde=True)
plt.show()
store['CompetitionDistance'].fillna(store['CompetitionDistance'].median(), inplace=True)

_ = store[pd.isnull(store.Promo2SinceWeek)]
_[_.Promo2 != 0].shape

store.fillna(0, inplace=True)

"""### 데이터 병합"""

train_store = pd.merge(train, store, how='inner', on='Store')

print("데이터 형태 : ", train_store.shape)
train_store.head()

"""### EDA

#### 가게 유형
"""

train_store.groupby('StoreType')['Sales'].describe()

train_store.groupby('StoreType')[['Customers', 'Sales']].sum()

train_store.columns

"""#### 가게 유형별 매출액 월별 추이"""
sns.catplot(data=train_store, x='Month', y="Sales",
            col='StoreType',
            palette='plasma',
            hue='StoreType',
            row='Promo',
            color=c)

plt.show()
"""#### 가게 유형별 고객 수 월별 추이"""

sns.catplot(data=train_store, x='Month', y="Customers",
            col='StoreType',
            palette='plasma',
            hue='StoreType',
            row='Promo',
            color=c)
plt.show()

"""#### 가게 유형별 고객 당 매출액 월별 추이"""

sns.catplot(data=train_store, x='Month', y="SalePerCustomer",
            col='StoreType',  # per store type in cols
            palette='plasma',
            hue='StoreType',
            row='Promo',  # per promo in the store in rows
            color=c)
plt.show()
"""#### 가게 유형별 매출액 주별 추이"""

# customers
sns.catplot(data=train_store, x='Month', y="Sales",
            col='DayOfWeek',  # per store type in cols
            palette='plasma',
            hue='StoreType',
            row='StoreType',  # per store type in rows
            color=c)
plt.show()
"""> 일요일에 문을 여는 가게들"""

train_store[(train_store.Open == 1) & (train_store.DayOfWeek == 7)]['Store'].unique()

train_store['CompetitionOpen'] = 12 * (train_store.Year - train_store.CompetitionOpenSinceYear) + \
                                 (train_store.Month - train_store.CompetitionOpenSinceMonth)

train_store['PromoOpen'] = 12 * (train_store.Year - train_store.Promo2SinceYear) + \
                           (train_store.WeekOfYear - train_store.Promo2SinceWeek) / 4.0

train_store.fillna(0, inplace=True)

train_store.loc[:, ['StoreType', 'Sales', 'Customers', 'PromoOpen', 'CompetitionOpen']].groupby('StoreType').mean()

"""### 상관성

#### 히트맵
"""

corr_all = train_store.drop('Open', axis=1).corr()

mask = np.zeros_like(corr_all, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(corr_all, mask=mask,
            square=True, linewidths=.5, ax=ax, cmap="BuPu")

plt.show()

"""#### 프로모션 조합별 매출액 주별 추이"""

sns.factorplot(data=train_store, x='DayOfWeek', y="Sales",
               col='Promo',
               row='Promo2',
               hue='Promo2',
               palette='RdPu')

"""## 계절성, 추세, 자기상관 기반 시계열 분석

### 계절성에 따른 매출액 특징
"""

train['Sales'] = train['Sales'] * 1.0

sales_a = train[train.Store == 2]['Sales']
sales_b = train[train.Store == 85]['Sales'].sort_index(ascending=True)
sales_c = train[train.Store == 1]['Sales']
sales_d = train[train.Store == 13]['Sales']

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 13))

sales_a.resample('W').sum().plot(color=c, ax=ax1)
sales_b.resample('W').sum().plot(color=c, ax=ax2)
sales_c.resample('W').sum().plot(color=c, ax=ax3)
sales_d.resample('W').sum().plot(color=c, ax=ax4)

"""### 연간 매출액 특징"""

f, (ax1, ax2, ax3, ax4) = plt.subplots(4, figsize=(12, 13))

decomposition_a = seasonal_decompose(sales_a, model='additive', freq=365)
decomposition_a.trend.plot(color=c, ax=ax1)

decomposition_b = seasonal_decompose(sales_b, model='additive', freq=365)
decomposition_b.trend.plot(color=c, ax=ax2)

decomposition_c = seasonal_decompose(sales_c, model='additive', freq=365)
decomposition_c.trend.plot(color=c, ax=ax3)

decomposition_d = seasonal_decompose(sales_d, model='additive', freq=365)
decomposition_d.trend.plot(color=c, ax=ax4)

"""### 자기상관 기반 매출액 추세"""

plt.figure(figsize=(12, 8))

# acf and pacf for A
plt.subplot(421)
plot_acf(sales_a, lags=50, ax=plt.gca(), color=c)
plt.subplot(422)
plot_pacf(sales_a, lags=50, ax=plt.gca(), color=c)

# acf and pacf for B
plt.subplot(423)
plot_acf(sales_b, lags=50, ax=plt.gca(), color=c)
plt.subplot(424)
plot_pacf(sales_b, lags=50, ax=plt.gca(), color=c)

# acf and pacf for C
plt.subplot(425)
plot_acf(sales_c, lags=50, ax=plt.gca(), color=c)
plt.subplot(426)
plot_pacf(sales_c, lags=50, ax=plt.gca(), color=c)

# acf and pacf for D
plt.subplot(427)
plot_acf(sales_d, lags=50, ax=plt.gca(), color=c)
plt.subplot(428)
plot_pacf(sales_d, lags=50, ax=plt.gca(), color=c)

plt.show()

"""## Prophet 을 활용한 매출 예측

> 오픈 가게에 대한 6주간의 매출액 예측
"""

df = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/CH16/train.csv",
                 low_memory=False)

df = df[(df["Open"] != 0) & (df['Sales'] != 0)]

# store number 1 (StoreType C)
sales = df[df.Store == 1].loc[:, ['Date', 'Sales']]

# reverse to the order: from 2013 to 2015
sales = sales.sort_index(ascending=False)

# to datetime64
sales['Date'] = pd.DatetimeIndex(sales['Date'])
sales.dtypes

# Prophet 에서는 모든 컬럼이 이름을 가져야 한다
sales = sales.rename(columns={'Date': 'ds',
                              'Sales': 'y'})
sales.head()

# 일일 매출 그래프
ax = sales.set_index('ds').plot(figsize=(12, 4), color=c)
ax.set_ylabel('Daily Number of Sales')
ax.set_xlabel('Date')
plt.show()

"""> 공휴일의 경우"""

# create holidays dataframe
state_dates = df[(df.StateHoliday == 'a') | (df.StateHoliday == 'b') & (df.StateHoliday == 'c')].loc[:, 'Date'].values
school_dates = df[df.SchoolHoliday == 1].loc[:, 'Date'].values

state = pd.DataFrame({'holiday': 'state_holiday',
                      'ds': pd.to_datetime(state_dates)})
school = pd.DataFrame({'holiday': 'school_holiday',
                       'ds': pd.to_datetime(school_dates)})

holidays = pd.concat((state, school))
holidays.head()

"""#### 모델링 및 예측"""

# Interval width : 95% (기본값 : 80%)
my_model = Prophet(interval_width=0.95,
                   holidays=holidays)
my_model.fit(sales)

# 6주 간의 미래 데이터를 만든다
future_dates = my_model.make_future_dataframe(periods=6 * 7)

print("첫번째 주 예측 : ")
future_dates.tail(7)

forecast = my_model.predict(future_dates)

# 마지막 주 예측
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

"""#### 시각화"""

my_model.plot(forecast)

my_model.plot_components(forecast)

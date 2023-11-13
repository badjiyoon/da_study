# -*- coding: utf-8 -*-
"""[전체]Ch17. [실습15] 이커머스 여름 의류 매출 분석.ipynb

> https://www.kaggle.com/jmmvutu/summer-products-and-sales-in-ecommerce-wish
* [summer-products-with-rating-and-performance_2020-08.csv]
* [unique-categories.sorted-by-count.csv]
* [unique-categories.csv]
#Part4. [실습15] 이커머스 여름 의류 매출 분석
"""
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
plt.rc('font', family='AppleGothic')

"""## 01. 데이터 소개 및 분석프로세스 수립
 : "강의자료 → Ch17. [실습15] 이커머스 여름 의류 매출 분석" 참고

## 02. 데이터 준비를 위한 EDA 및 전처리
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 43)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 200)
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

import plotly.io as pio

pio.renderers.default = 'colab'
pio.renderers

import warnings

warnings.filterwarnings("ignore")

import os

"""### 0. 데이터 불러오기"""

df = pd.read_csv(
    '/Users/jiyoonkim/Documents/da_study/comFiles/CH17/summer-products-with-rating-and-performance_2020-08.csv')  #
df

"""### 데이터 타입"""

# 데이터 컬럼 이름/타입 정보 확인하기
print(df.info())

# 데이터 타입별 컬럼 수 확인하기
dtype_data = df.dtypes.reset_index()
dtype_data.columns = ["Count", "Column Type"]
dtype_data = dtype_data.groupby("Column Type").aggregate('count').reset_index()

print(dtype_data)

"""### 통계값"""

round(df.describe())

"""### 결측값"""

df.isnull().sum()


def plot_missing_data(df):
    columns_with_null = df.columns[df.isna().sum() > 0]
    null_pct = (df[columns_with_null].isna().sum() / df.shape[0]).sort_values(ascending=False) * 100
    plt.figure(figsize=(8, 6))
    sns.barplot(y=null_pct.index, x=null_pct, orient='h')
    plt.title('% Na values in dataframe by columns')


plot_missing_data(df)
plt.show()
"""### 중복값"""

# 중복된 항목 수 알아보기
print("중복된 항목 수 :", len(df[df.duplicated()]))

# 중복된 항목 확인
print(df[df.duplicated(keep=False)].sort_values(by=list(df.columns)).head())

# 중복된 항목 제거
df.drop_duplicates(inplace=True, keep='first', ignore_index=True)

"""### EDA

> 히스토그램 / 카운트플롯 그리기
"""


def histograms_and_countplots(column, data, columns_to_exclude):
    if column not in columns_to_exclude:
        if data[column].dtype not in ['int64', 'float64']:
            f, axes = plt.subplots(1, 1, figsize=(15, 5))
            sns.countplot(x=column, data=data)
            plt.xticks(rotation=90)
            plt.suptitle(column, fontsize=20)
            plt.show()
        else:
            g = sns.FacetGrid(data, margin_titles=True, aspect=4, height=3)
            g.map(plt.hist, column, bins=100)
            plt.show()
        plt.show()


columns_to_exclude = ['title', 'title_orig', 'currency_buyer', 'tags', 'merchant_title', 'merchant_name',
                      'merchant_info_subtitle', 'merchant_id', 'merchant_profile_picture', 'product_url',
                      'product_picture', 'product_id', 'theme', 'crawl_month']

for column in df.columns:
    histograms_and_countplots(column, df, columns_to_exclude)

"""> 판매 현황 및 인기 상품

* price 와 retail_price
"""

df.describe().T

plt.figure(figsize=(12, 6))
sns.distplot(df['price'], color='red', label='Price')
sns.distplot(df['retail_price'], color='blue', label='Retail price')
plt.legend()
plt.show()

fig = go.Figure()
fig.add_trace(go.Box(x=df['retail_price'], name='Retail Price'))
fig.add_trace((go.Box(x=df['price'], name='Price')))
fig['layout']['title'] = 'Distribution of Price and Retail Price'
fig.show()

"""> Units Sold 데이터"""

print('단위 판매량 Median : ', df['units_sold'].median())
print('단위 판매량 Mean : ', df['units_sold'].mean())
df['units_sold'].value_counts()

"""> 10개 미만 올림"""


def below_ten(units_sold):
    if units_sold < 10:
        return 10
    else:
        return units_sold


df['units_sold'] = df['units_sold'].apply(below_ten)

df['units_sold'].value_counts()

fig_dims = (10, 15)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(data=df,
              order=df['units_sold'].value_counts().index,
              ax=ax,
              x='units_sold')
ax.set(xlabel='Units Sold', ylabel='Count')
fig.show()

px.scatter(df, x='units_sold', y='price', marginal_x='box', title='Price vs Units Sold')
"""> 인기 상품 기준을 1000개로 라벨 추가"""


def is_successful(units_sold):
    if units_sold > 1000:
        return 1
    else:
        return 0


df['is_successful'] = df['units_sold'].apply(is_successful)
print('Percent of successful products: ', df['is_successful'].value_counts()[1] / len(df['is_successful']) * 100)
sns.countplot(data=df, x='is_successful')
plt.show()

"""`**Units Sold**` 기준으로 Clustering, 판매량 예측 모델링

## K-Means Clustering 을 활용한 판매량 영향 인자 분석

### 데이터 확인
"""

# range for units sold
# 판매량
sorted(df['units_sold'].unique())

"""### Elbow Method"""
from sklearn.cluster import KMeans

clusters = {}
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i).fit(df[['units_sold']])
    clusters[i] = kmeans.inertia_

plt.plot(list(clusters.keys()), list(clusters.values()));
plt.xlabel('no. of clusters')
plt.ylabel('kmeans inertia')
plt.show()

# order cluster method
def order_cluster(cluster_field_name, target_field_name, df, ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name, ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df, df_new[[cluster_field_name, 'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name], axis=1)
    df_final = df_final.rename(columns={"index": cluster_field_name})
    return df_final


df['units_sold_cluster'] = KMeans(n_clusters=3).fit(df[['units_sold']]).predict(df[['units_sold']])
df = order_cluster('units_sold_cluster', 'units_sold', df, True)
df.groupby(['units_sold_cluster'])['units_sold'].describe()

"""### 상관성에 따른 클러스터 분석"""
# number인것만 처리해도 됌
features = ['price', 'retail_price', 'units_sold', 'rating', 'rating_count', 'shipping_option_price',
            'product_variation_inventory', 'merchant_rating', 'merchant_rating_count']
corr = df[features].corr(method='spearman')

plt.figure(figsize=(15, 8))
sns.heatmap(corr, annot=True)
plt.show()

px.scatter(df, x='units_sold', y='rating', color='units_sold_cluster', marginal_y='box', title='Rating vs units sold')

px.scatter(df, x='rating', y='merchant_rating', color='units_sold_cluster', marginal_y='box',
           title='Merchant Rating vs units sold', opacity=0.7)

px.scatter(df, x='rating', y='product_variation_inventory', color='units_sold_cluster',
           title='Product variation vs Rating')

fig = px.scatter(df, x='rating_count', y='rating', color='units_sold_cluster', title='Rating vs Rating count')
fig.update_layout(showlegend=False)

px.scatter(df, x='retail_price', y='price', color='units_sold_cluster', marginal_y='box')

px.scatter(df, x='price', y='shipping_option_price', color='units_sold_cluster', title='Shipping price vs Price')

"""### 할인율과 클러스터"""

df['difference'] = df['retail_price'] - df['price']
df['discount'] = df['difference'] / df['retail_price'] * 100
plt.figure(figsize=(12, 6))
sns.distplot(df['discount']);
plt.title('Distribution of Discount');
plt.show()
px.scatter(df, x='discount', y='rating_count', color='units_sold_cluster')

"""## 다양한 Classifier 를 활용한 판매량 예측

### 데이터 정제
"""

salesData = df.drop(
    ['is_successful', 'units_sold_cluster', 'crawl_month', 'product_id', 'product_picture', 'product_url',
     'merchant_profile_picture', 'merchant_id', 'currency_buyer'], axis=1)

salesData = salesData.drop(['theme', 'urgency_text', 'merchant_title', 'merchant_name', 'merchant_info_subtitle'],
                           axis=1)
salesData = salesData.drop(['title', 'title_orig', 'tags'], axis=1)
salesData = salesData.drop(['shipping_option_name'], axis=1)
salesData = salesData.drop(['rating_count'], axis=1)

salesData.head()

salesData.info()

"""### Target Feature 정의

#### 유일한 값
"""

salesData['units_sold'].unique()
salesData['units_sold'].value_counts()

"""### 데이터 타입별 Feature 전환

#### 데이터 타입별 컬럼 수
"""

# 데이터 타입별 컬럼 수 확인
dtype_data = salesData.dtypes.reset_index()
dtype_data.columns = ["Count", "Column Type"]
dtype_data = dtype_data.groupby("Column Type").aggregate('count').reset_index()

print(dtype_data)

"""#### 범주형 Feature"""

# pandas 의 select_dtypes('object') 사용
salesData.select_dtypes(include=['object', 'category']).head()

cat_feat = salesData.select_dtypes('object', 'category').columns.values
salesData_cat = salesData[cat_feat].copy()
print(salesData_cat.nunique().sort_values())

salesData_cat_dummies = pd.get_dummies(salesData_cat)
salesData_cat_dummies.head(3)

"""#### 숫자형 Feature"""

# pandas 의 select_dtypes('number') 사용
salesData.select_dtypes(include=['number']).head()

num_feat = salesData.select_dtypes('number').columns.values
salesData_num = salesData[num_feat].copy()

print(salesData_num.nunique().sort_values())

"""> 유일한 값이 1개인 경우 또는 모든 행의 값이 다른 경우는 제거한다
* Target Feature 가 포함되어 있으면 함께 제거
"""
salesData_num = salesData_num.drop(['has_urgency_banner', 'units_sold'], axis=1, errors='ignore')
display(salesData_num.isnull().any())
salesData_num.fillna(0, inplace=True)

"""### 상관성에 따른 Feature 정제"""
from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations

num_feat = salesData_num.columns.values
comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(salesData_num[comb[0]], salesData_num[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)

high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.9]
high_corr_num

salesData_num = salesData_num.drop(
    ['difference', 'rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count',
     'rating_one_count'], axis=1, errors='ignore')

salesData_num.info()

"""### 모델링

#### 최종 데이터 확인
"""
train_final = pd.concat([salesData['units_sold'], salesData_num, salesData_cat_dummies], axis=1)
train_final.head()

from sklearn.model_selection import train_test_split

r_state = 3
X = train_final.drop(['units_sold'], axis=1)
y = train_final['units_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=r_state)

"""#### DecisionTreeClassifier"""
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

classifier_DTC = DecisionTreeClassifier(random_state=r_state)
classifier_DTC.fit(X_train, y_train)

y_pred_DTC = classifier_DTC.predict(X_test)
accuracy_score(y_test, y_pred_DTC)

"""#### SVC"""
from sklearn.svm import SVC

classifier_SVC = SVC(random_state=r_state)
classifier_SVC.fit(X_train, y_train)

y_pred_SVC = classifier_SVC.predict(X_test)
accuracy_score(y_test, y_pred_SVC)

"""#### AdaBoostClassifier"""
from sklearn.ensemble import AdaBoostClassifier
classifier_ABC = AdaBoostClassifier(learning_rate=0.01,
                                    random_state=r_state)  # Default using Decision Tree Classifier
classifier_ABC.fit(X_train, y_train)

y_pred_ABC = classifier_ABC.predict(X_test)
accuracy_score(y_test, y_pred_ABC)

"""#### RandomForestClassifier"""
from sklearn.ensemble import RandomForestClassifier

classifier_RFC = RandomForestClassifier(random_state=r_state)
classifier_RFC.fit(X_train, y_train)

y_pred_RFC = classifier_RFC.predict(X_test)
accuracy_score(y_test, y_pred_RFC)

"""#### AdaBoost - Random Forest"""

classifier_ABC_RF = AdaBoostClassifier(RandomForestClassifier(random_state=r_state),
                                       learning_rate=0.01,
                                       random_state=r_state)
classifier_ABC_RF.fit(X_train, y_train)

y_pred_ABC_RF = classifier_ABC_RF.predict(X_test)
accuracy_score(y_test, y_pred_ABC_RF)

"""#### GradientBoostingClassifier"""

from sklearn.ensemble import GradientBoostingClassifier

classifier_GBC = GradientBoostingClassifier(random_state=r_state)
classifier_GBC.fit(X_train, y_train)

y_pred_GBC = classifier_GBC.predict(X_test)
accuracy_score(y_test, y_pred_GBC)

"""#### KNeighborsClassifier"""

from sklearn.neighbors import KNeighborsClassifier

classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train, y_train)

y_pred_KNN = classifier_KNN.predict(X_test)
accuracy_score(y_test, y_pred_KNN)

"""#### XGBClassifier"""
from xgboost import XGBClassifier

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

classifier_XGB = XGBClassifier()
classifier_XGB.fit(X_train, y_train)

y_pred_XGB = classifier_XGB.predict(X_test)
accuracy_score(y_test, y_pred_XGB)

from sklearn.model_selection import cross_val_score

classifiers = [classifier_DTC,
               classifier_SVC,
               classifier_ABC,
               classifier_RFC,
               classifier_ABC_RF,
               classifier_GBC,
               classifier_KNN,
               classifier_XGB]
classifiers_names = ['Decision Tree',
                     'SVC',
                     'AdaBoost',
                     'Random Forest',
                     'AdaBoost - Random Forest',
                     'Gradient Boosting',
                     'KNeighborsClassifier',
                     'XG Boost']
accuracy_mean = []

for cl in classifiers:
    accuracies = cross_val_score(estimator=cl,
                                 X=X_train,
                                 y=y_train,
                                 cv=4)
    accuracy_mean.append(accuracies.mean() * 100)

accuracy_df = pd.DataFrame({'Classifier': classifiers_names,
                            'Accuracy Mean': accuracy_mean})
accuracy_df.sort_values('Accuracy Mean', ascending=False)

"""#### VotingClassifier"""

from sklearn.ensemble import VotingClassifier

voting_cl = VotingClassifier(estimators=[('Gradient Boosting', classifier_GBC),
                                         ('Decision Tree', classifier_RFC),
                                         ('XG Boost', classifier_DTC)],
                             voting='hard')
voting_cl.fit(X_train, y_train)
y_pred_vcl = voting_cl.predict(X_test)
accuracy_score(y_test, y_pred_vcl)

"""#### GridSearchCV 적용해보기"""

from sklearn.model_selection import GridSearchCV

gb_params = [{'loss': ['deviance', 'exponential'],
              'learning_rate': [0.1, 0.01, 0.001],
              'n_estimators': [100, 250, 500]}]

grid_search_GBC = GridSearchCV(estimator=classifier_GBC,
                               param_grid=gb_params,
                               scoring='accuracy',
                               cv=4,
                               n_jobs=-1)

grid_search_GBC.fit(X_train, y_train)
best_accuracy_GBC = grid_search_GBC.best_score_
best_parameters_GBC = grid_search_GBC.best_params_
print("Best Accuracy: {:.2f} %".format(best_accuracy_GBC * 100))
print("Best Parameters:", best_parameters_GBC)

optimised_GBC = GradientBoostingClassifier(random_state=r_state,
                                           loss='deviance',
                                           learning_rate=0.1,
                                           n_estimators=500)
optimised_GBC.fit(X_train, y_train)
y_pred_GBC = optimised_GBC.predict(X_test)
accuracy_score(y_test, y_pred_GBC)

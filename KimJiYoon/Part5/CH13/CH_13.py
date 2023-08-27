# * amazon.in 의 COVID-19 관련 이커머스 상품 웹 크롤링 데이터
# * 예제) https://www.amazon.in/Aurum-Creations-Disposable-Stretchable-100/dp/B0777HKWJF/
# Part4. [실습11] COVID-19 관련 이커머스 상품 판매 데이터 분석
# 인도 아마존 데이터
from matplotlib import pyplot as plt

plt.rc('font', family='AppleGothic')

# 구매 선호도 예측 모델 생성
# 01. 데이터 소개 및 분석프로세스 수립
#  : "강의자료 → Ch13. [실습11] COVID-19 관련 이커머스 상품 판매 데이터 분석" 참고
# 02. 데이터 준비를 위한 EDA 및 전처리
# 0. 데이터 불러오기
############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import pandas as pd
import numpy as np

np.bool = np.bool_
import seaborn as sns
import matplotlib.pyplot as plt

import sys
import plotly.express as px
from numpy import random
import missingno

from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

import pydot
from IPython.display import Image

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 20)  ## 모든 열을 출력한다.
pd.set_option('display.max_rows', 20)  ## 모든 행을 출력한다.

amz_data = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/amz_data.csv')
amz_data.head()

amz_data = amz_data.drop(['index'], axis=1, errors='ignore')
amz_data.head()

# 데이터의 모양 알아보기
amz_data.shape

# 1. 데이터 탐색
# 1) 데이터 타입
# 컬럼별 데이터 타입 알아보기
amz_data.info()
# 2) 데이터 통계값
# 컬럼별 간단한 통계값 보기
amz_data.describe()
# 3) 결측값
missing_df = amz_data.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df

# seaborn 패키지 heatmap 을 통해 시각화 확인
sns.heatmap(amz_data.isnull(), cbar=False, yticklabels=False, cmap='viridis')
plt.show()
# missingno 패키지를 통해 확인
missingno.matrix(amz_data, figsize=(30, 10))
plt.show()

# num_cols : 숫자형 컬럼의 총 개수
num_cols = amz_data.select_dtypes(include=np.number).shape[1]

# 숫자형 컬럼이 모두 NULL 인 행은 삭제
amz_data = amz_data[amz_data.select_dtypes(include=np.number).isnull().sum(axis=1) != num_cols]
# 4) 중복값
# 중복된 항목 수 알아보기
print("중복된 항목 수 :", len(amz_data[amz_data.duplicated()]))
# 중복된 항목 확인
amz_data[amz_data.duplicated(keep=False)].sort_values(by=list(amz_data.columns))
# 중복된 항목 제거
amz_data.drop_duplicates(inplace=True, keep='first', ignore_index=True)

# 5) 변수별 시각화
# > 분포
# 데이터 컬럼 타입이 np.number 인 것만 가져오기
numeric_data = amz_data.select_dtypes(include=np.number)

# 데이터 컬럼 타입이 np.number 인 컬럼 이름들 가져오기
l = numeric_data.columns.values
number_of_columns = 4
number_of_rows = len(l) - 1 / number_of_columns

# 컬럼별 히스토그램 그리기
for i in range(0, len(l)):
    target_data = numeric_data[l[i]]

    # 결측값에 채운 0 을 제외한 데이터 가져오기
    target_data_wo_zero = target_data[target_data > 0]
    sns.displot(target_data_wo_zero, kde=True)  # kde : kernel density
    plt.show()

# 데이터 컬럼 타입이 np.number 인 컬럼들 가져오기
columns = amz_data.select_dtypes(include=np.number).columns
figure = plt.figure(figsize=(20, 10))
figure.add_subplot(1, len(columns), 1)
for index, col in enumerate(columns):
    if index > 0:
        figure.add_subplot(1, len(columns), index + 1)
    sns.boxplot(y=col, data=amz_data, boxprops={'facecolor': 'None'})
figure.tight_layout()  # 자동으로 명시된 여백에 관련된 서브플롯 파라미터를 조정한다.
plt.show()
# ### 3. 데이터 타입별 Feature 변환
# #### 1) Feature 탐색
# ##### 총 Feature 개수 확인
print(amz_data.info())
# ##### Feature 데이터 타입별 개수 확인
# 데이터 타입별 컬럼 수 확인
dtype_data = amz_data.dtypes.reset_index()
dtype_data.columns = ["Count", "Column Type"]
dtype_data = dtype_data.groupby("Column Type").aggregate('count').reset_index()

print(dtype_data)
# #### 3) 숫자형 Feature
#   * 데이터 확인
#   * Feature 제거
# ##### 데이터 확인
# pandas 의 select_dtypes('object') 사용
amz_data.select_dtypes(include=['number']).head()
num_feat = amz_data.select_dtypes('number').columns.values
train_num = amz_data[num_feat].copy()
# Feature 제거
# > Feature 별 유일한 값 개수 확인
print(train_num.nunique().sort_values())
print(amz_data.shape)
# > 유일한 값이 1개인 경우 또는 모든 행의 값이 다른 경우는 제거한다
# ## 03. 제품별 군집분석
# #### 1) K-Means Clustering
# ##### Scailing 과 Elbow 방법 적용
amz_all = amz_data
amz_all = amz_all.reset_index()
prd_lists = list(amz_all['product'].unique())
prd_lists

scaler = StandardScaler()
valid_prd_list = []

for prd in prd_lists:
    df_scaled = scaler.fit_transform(
        amz_all[['mrp_then', 'price_then', 'price_now', 'mrp_now', 'star5', "star4", 'star3', 'star2', 'star1']][
            amz_all['product'] == prd])
    df_scaled = np.nan_to_num(df_scaled)
    wcss = []
    try:
        for i in range(1, 19):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(df_scaled)
            wcss.append(kmeans.inertia_)

        plt.title(prd)
        plt.plot(wcss, 'bx-')
        plt.show()
        valid_prd_list.append(prd)
    except:
        print(prd + ' : ' + str(sys.exc_info()))

# 2) 제품-클러스터별 변수 분포도
columnsss_amz = ['mrp_then', 'price_then', 'price_now', 'mrp_now', 'star5', "star4", 'star3', 'star2', 'star1']
kmeans_dict_amz = {'medical equipment': 7, 'Skin Care': 5, 'facemasks': 7}
labels_dict_amz = {}

for prd in list(valid_prd_list):
    print(prd)
    df_scaled = scaler.fit_transform(amz_all[columnsss_amz][amz_all['product'] == prd])
    df_scaled = np.nan_to_num(df_scaled)
    kmeans = KMeans(kmeans_dict_amz[prd])
    kmeans.fit(df_scaled)
    labels = kmeans.labels_
    labels_dict_amz[prd] = labels
    cluster_centers = pd.DataFrame(data=kmeans.cluster_centers_, columns=columnsss_amz)
    df_cluster = pd.concat([amz_all[columnsss_amz][amz_all['product'] == prd], pd.DataFrame({'cluster': labels})],
                           axis=1)
    for j in columnsss_amz:
        plt.figure(figsize=(35, 5))
        for k in range(kmeans_dict_amz[prd]):
            plt.subplot(1, kmeans_dict_amz[prd], k + 1)
            cluster = df_cluster[df_cluster['cluster'] == k]
            cluster[j].hist(bins=20)
            plt.title(prd + ' {}    \nCluster {} '.format(j, k))

        plt.show()

amz_all['product']
# 3) 클러스터-PCA 분석
palette = ['red', 'green', 'blue', 'pink', 'yellow', 'gray', 'purple', 'black', 'magenta', 'cyan']
for prd in valid_prd_list:
    print(prd)
    df_scaled = scaler.fit_transform(
        amz_all[['mrp_then', 'price_then', 'price_now', 'mrp_now', 'star5', "star4", 'star3', 'star2', 'star1']][
            amz_all['product'] == prd])
    df_scaled = np.nan_to_num(df_scaled)

    pca = PCA(n_components=2)
    principal_comp = pca.fit_transform(df_scaled)
    pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
    pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels_dict_amz[prd]})], axis=1)

    plt.figure(figsize=(10, 10))
    ax = sns.scatterplot(x="pca1", y="pca2", hue="cluster", data=pca_df, palette=palette[:kmeans_dict_amz[prd]])
    plt.title(prd)
    plt.show()

# ## 04. 구매 선호도 예측 분석
# ### 1) Target Feature 생성
buyability = []
for i in range(amz_all.shape[0]):
    averaged_score = (amz_all['star1'][i] * (1 / 15)) + (amz_all['star2'][i] * (2 / 15)) + (
            amz_all['star3'][i] * (3 / 15)) + (amz_all['star4'][i] * (4 / 15)) + (amz_all['star5'][i] * (5 / 15))
    try:
        averaged_score = int(averaged_score)
    except:
        averaged_score = 0
    buyability.append(averaged_score)

amz_all['buyability'] = buyability
# ### 2) 결측값 처리
missingno.matrix(amz_all, figsize=(30, 10))
amz_all.fillna(0, inplace=True)
missingno.matrix(amz_all, figsize=(30, 10))
# ### 3) 데이터 준비
X = amz_all[amz_all.columns[2:-1]]
y = amz_all['buyability']
X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=85)
X_train = X_train.drop(['name'], axis=1, errors='ignore')
x_test = x_test.drop(['name'], axis=1, errors='ignore')

y_test
# ### 4) Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(x_test)
print(accuracy_score(y_test, y_pred))
# ### 5) SVM
# > Scailing
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train[:5, :]

clf = svm.SVC(kernel='sigmoid')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print(accuracy_score(y_train, y_pred))

# > 다양한 SVM Kernel 모델링

for k in ('linear', 'poly', 'rbf', 'sigmoid'):
    clf = svm.SVC(kernel=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    print(k)
    print(accuracy_score(y_train, y_pred))

# > Best Model

clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)

X_test = scaler.transform(x_test)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

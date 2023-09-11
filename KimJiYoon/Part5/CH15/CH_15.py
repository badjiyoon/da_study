# -*- coding: utf-8 -*-
"""[전체]Ch15. [실습13] 소매 판매 데이터를 활용한 이커머스 고객 Segmentation 분석.ipynb
* [data.csv] : https://www.kaggle.com/carrie1/ecommerce-data
#Part4. [실습13] 소매 판매 데이터를 활용한 이커머스 고객 Segmentation 분석
## 01. 데이터 소개 및 분석프로세스 수립
 : "강의자료 → Ch04. [실습13] 소매 판매 데이터를 활용한 이커머스 고객 Segmentation 분석" 참고
## 02. 데이터 탐색 및 전처리
"""
from matplotlib import pyplot as plt

plt.rc('font', family='AppleGothic')

### 0. 데이터 불러오기
# Commented out IPython magic to ensure Python compatibility.
############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import pandas as pd

pd.set_option('display.max_columns', 8)
pd.set_option('display.max_rows', 40)
pd.set_option('display.width', 200)
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import datetime, nltk, warnings
import matplotlib.cm as cm
import itertools

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from IPython.display import display, HTML
# import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
warnings.filterwarnings("ignore")
plt.rcParams["patch.force_edgecolor"] = True
plt.style.use('fivethirtyeight')
mpl.rc('patch', edgecolor='dimgray', linewidth=1)

# ID의 경우 스트링으로 변환하여 지정 처리
df_init = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/data.csv', encoding="ISO-8859-1",
                      dtype={'CustomerID': str, 'InvoiceID': str})
print('Dataframe dimensions:', df_init.shape)

"""### 1. 데이터 준비"""
# Description안에만 판 물건에 대한 정보가 있음
df_init.head(5)

df_init.dtypes

# null 데이터 CustomerID, Description
df_init.isnull().sum()

df_init.shape[0]

# 날짜 데이터 변환 -> String -> Date 타입으로 변환 처리
df_init['InvoiceDate'] = pd.to_datetime(df_init['InvoiceDate'])

# 데이터 타입 정보 T-> 가로로 변경 Transpose 맨앞에 인덱스를 column type을 붙여줌
col_info = pd.DataFrame(df_init.dtypes).T.rename(index={0: 'column type'})
display(col_info)

"""#### 1) 컬럼별 데이터 타입 및 결측값 비율"""
# 정보를 쌓는 형태로 표현함.
# null 카운트
col_info = pd.concat([col_info, pd.DataFrame(df_init.isnull().sum()).T.rename(index={0: 'null values (nb)'})])
display(col_info)

# %비율로 나눠서 추가 처리
col_info = pd.concat(
    [col_info, pd.DataFrame(df_init.isnull().sum() / df_init.shape[0] * 100).T.rename(index={0: 'null values (%)'})])
display(col_info)

display(df_init[:5])

"""#### 2) 결측값 처리"""
df_init.dropna(axis=0, subset=['CustomerID'], inplace=True)
print('Shape:', df_init.shape)

col_info = pd.DataFrame(df_init.dtypes).T.rename(index={0: 'column type'})
col_info = pd.concat([col_info, pd.DataFrame(df_init.isnull().sum()).T.rename(index={0: 'null values (nb)'})])
col_info = pd.concat(
    [col_info, pd.DataFrame(df_init.isnull().sum() / df_init.shape[0] * 100).T.rename(index={0: 'null values (%)'})])
display(col_info)

df_init.dtypes

df_init.isnull().sum()

"""#### 3) 중복값"""
df_init['Country'].duplicated().value_counts()
# 같은 송장에 여라가지 제품을 판것을 확인할 수 있음
df_init['InvoiceNo'].duplicated().value_counts()

# 전체 중복의 경우 중복 데이터 삭제
print('Duplicate Entries: {}'.format(df_init.duplicated().sum()))
df_init.drop_duplicates(inplace=True)

"""#### 4) 나라별 지도 시각화"""
# 카운트가 없음
temp_cou = df_init[['CustomerID', 'InvoiceNo', 'Country']].groupby(['CustomerID', 'InvoiceNo', 'Country']).count()
temp_cou
# reset_index를 함으로써 위에 작업한부분이 데이터화가 됨
temp_cou = temp_cou.reset_index(drop=False)
temp_cou

countries = temp_cou['Country'].value_counts()
print('No. of countries in the dataframe: {}'.format(len(countries)))

"""Let's display the result on a chloropleth map"""
countries.index
countries

"""A `choropleth map` is a type of thematic map in which areas are shaded or patterned in proportion to a statistical variable that represents an aggregate summary of a geographic characteristic within each area, such as population density or per-capita income."""

import plotly.io as pio

pio.renderers.default = 'notebook_connected'
pio.renderers
pio.renderers.default = 'colab'
pio.renderers

# 나라별 주문의 수 시각하 하기
data = dict(type='choropleth',
            locations=countries.index,
            locationmode='country names', z=countries,
            text=countries.index, colorbar={'title': 'Order no.'},
            colorscale=[[0, 'rgb(224,255,255)'],
                        [0.01, 'rgb(166,206,227)'], [0.02, 'rgb(31,120,180)'],
                        [0.03, 'rgb(178,223,138)'], [0.05, 'rgb(51,160,44)'],
                        [0.10, 'rgb(251,154,153)'], [0.20, 'rgb(255,255,0)'],
                        [1, 'rgb(227,26,28)']],
            reversescale=False)

layout = dict(title='Number of orders per country',
              geo=dict(showframe=True, projection={'type': 'mercator'}))

choromap = go.Figure(data=[data], layout=layout)
iplot(choromap, validate=False)

df_init

"""> 각 항목별 데이터 개수"""
# 각 독립된 항목의 수
len(df_init['CustomerID'].value_counts())
pd.DataFrame([{'products': len(df_init['StockCode'].value_counts()),
               'transactions': len(df_init['InvoiceNo'].value_counts()),
               'customers': len(df_init['CustomerID'].value_counts()), }],
             columns=['products', 'transactions', 'customers'], index=['quantity'])

"""> 고객 주문번호별 상품 개수"""
temp_pro = df_init.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate'].count()
nb_products_per_basket = temp_pro.rename(columns={'InvoiceDate': 'Number of products'})
nb_products_per_basket[:10].sort_values('CustomerID')

"""> 고객 주문번호별 주문 취소 여부와 주문 취소율"""
# InvoiceNo -> c인경우 주문 취소를 의미한다.
nb_products_per_basket['order_canceled'] = nb_products_per_basket['InvoiceNo'].apply(lambda x: int('C' in x))
display(nb_products_per_basket[:5])

n1 = nb_products_per_basket['order_canceled'].sum()

n2 = nb_products_per_basket.shape[0]

print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1 / n2 * 100))

# Quantity 항목이 -로 처리되어있음
display(df_init.sort_values('CustomerID')[:5])

"""#### 5) 주문 취소된 항목 삭제한 새로운 컬럼 만들기"""
df_cleaned = df_init.copy(deep=True)
df_cleaned['QuantityCanceled'] = 0

# 삭제할 엔트리
entry_to_remove = []
# 이상 엔트리
doubtfull_entry = []

for index, col in df_init.iterrows():
    # 주문 수량이 0보다 크거나 할인의 경우
    # Discount의 경우 주문 취소가 되지 않음
    if (col['Quantity'] > 0) or col['Description'] == 'Discount':
        continue
    # 테스트 데이터를 만듬 주문데이터의 후보군을 만듬
    df_test = df_init[(df_init['CustomerID'] == col['CustomerID']) &
                      (df_init['StockCode'] == col['StockCode']) &
                      (df_init['InvoiceDate'] < col['InvoiceDate']) &
                      (df_init['Quantity'] > 0)].copy()

    # 주문일이 더 빠르고 수량이 0보다 큰 경우가 1개도 없는 경우
    if (df_test.shape[0] == 0):
        doubtfull_entry.append(index)

    # 주문일이 더 빠르고 수량이 0보다 큰 경우가 1개 있는 경우
    elif (df_test.shape[0] == 1):
        index_order = df_test.index[0]
        # 취소된 수량에 대한 컬럼을 새로 생성한다.
        df_cleaned.loc[index_order, 'QuantityCanceled'] = -col['Quantity']
        entry_to_remove.append(index)

    # 주문일이 더 빠르고 수량이 0보다 큰 경우가 2개 이상인 경우
    elif (df_test.shape[0] > 1):
        df_test.sort_index(axis=0, ascending=False, inplace=True)

        for ind, val in df_test.iterrows():
            if val['Quantity'] < -col['Quantity']: continue
            df_cleaned.loc[ind, 'QuantityCanceled'] = -col['Quantity']
            entry_to_remove.append(index)
            break

print("entry_to_remove: {}".format(len(entry_to_remove)))
print("doubtfull_entry: {}".format(len(doubtfull_entry)))

df_cleaned.drop(entry_to_remove, axis=0, inplace=True)
df_cleaned.drop(doubtfull_entry, axis=0, inplace=True)

# StockCode D의 경우 할인 항목 취소 불가
remaining_entries = df_cleaned[(df_cleaned['Quantity'] < 0) & (df_cleaned['StockCode'] != 'D')]
print("nb of entries to delete: {}".format(remaining_entries.shape[0]))
remaining_entries.head(5)

remaining_entries.sort_index(axis=0)[:5]

df_cleaned.head(5)

"""#### 6) StockCode 내 데이터 수정"""

df_cleaned.info()

list_special_codes = df_cleaned[df_cleaned['StockCode'].str.contains('^[a-zA-Z]+', regex=True)]['StockCode'].unique()
list_special_codes

# Stock Code Descrition을 맵핑해본 결과
for code in list_special_codes:
    print("{:<15} -> {:<30}".format(code, df_cleaned[df_cleaned['StockCode'] == code]['Description'].unique()[0]))

"""#### 7) 구매액 컬럼 생성"""
df_cleaned['TotalPrice'] = df_cleaned['UnitPrice'] * (df_cleaned['Quantity'] - df_cleaned['QuantityCanceled'])
df_cleaned.sort_values('CustomerID')[:5]

df_cleaned.info()

"""#### 8) 주문일 컬럼 데이터타입 변경"""
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
df_cleaned[:5]

"""#### 9) 고객 주문번호별 총 구매액/평균 주문시기 데이터 생성"""
temp_sum = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp_sum.rename(columns={'TotalPrice': 'Basket Price'})

# 평균 주문 시기
df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp_date = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp_date['InvoiceDate_int'])

basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID')[:6]

basket_price.tail(6)

basket_price.max(), basket_price.min()

"""#### 10) 키워드 생성"""
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
df_products = pd.DataFrame(df_init['Description'].unique()).rename(columns={0: 'Description'})
df_products

"""> 데이터 준비를 위한 자연어처리 내용 소개"""
# 명사 관련 처리 예졔
is_noun = lambda pos: pos[:2] == 'NN'
ex = "WHITE HANGING HEART T-LIGHT HOLDER"
lines = ex.lower()
tokenized = nltk.word_tokenize(lines)
nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

lines

tokenized

# POS(part-of-speech)는 품사를 말한다.
nltk.pos_tag(tokenized)

nouns

# stemming : 어간(활용어에서 변하지 않는 부분)추출 - 어형이 변형된 단어로부터 접사 등을 제거하고 그 단어의 어간을 분리해 내는 것
stemmer = nltk.stem.SnowballStemmer("english")
print("The stemmed form of studying is: {}".format(stemmer.stem("studying")))
print("The stemmed form of studies is: {}".format(stemmer.stem("studies")))
print("The stemmed form of study is: {}".format(stemmer.stem("study")))

t = "heart"
racine = stemmer.stem(t)
racine

is_noun = lambda pos: pos[:2] == 'NN'


def keywords_inventory(dataframe, colonne='Description'):
    stemmer = nltk.stem.SnowballStemmer("english")

    keywords_roots = dict()
    keywords_select = dict()
    category_keys = []
    count_keywords = dict()
    icount = 0

    # dataframe[colonne] : Description
    for s in dataframe[colonne]:
        if pd.isnull(s):
            continue

        # Description 내 1개 데이터에 대해 소문자 처리
        lines = s.lower()

        # nltk 의 word_tokenize :
        tokenized = nltk.word_tokenize(lines)
        nouns = [word for (word, pos) in nltk.pos_tag(tokenized) if is_noun(pos)]

        for t in nouns:
            t = t.lower()
            racine = stemmer.stem(t)

            if racine in keywords_roots:
                keywords_roots[racine].add(t)
                count_keywords[racine] += 1
            else:
                # 어간을 key 로 tokenizing 해서 얻은 명사(NN)을 value 로 저장해 준다.
                keywords_roots[racine] = {t}
                count_keywords[racine] = 1

    for s in keywords_roots.keys():
        # 같은 어간을 가지는 것들이 2개 이상일 때
        if len(keywords_roots[s]) > 1:
            min_length = 1000
            for k in keywords_roots[s]:
                if len(k) < min_length:
                    clef = k
                    min_length = len(k)
            category_keys.append(clef)
            keywords_select[s] = clef
        else:
            category_keys.append(list(keywords_roots[s])[0])
            keywords_select[s] = list(keywords_roots[s])[0]

    print("'{}' 컬럼 안의 Keyword 수 : {}".format(colonne, len(category_keys)))
    return category_keys, keywords_roots, keywords_select, count_keywords


keywords, keywords_roots, keywords_select, count_keywords = keywords_inventory(df_products)

keywords_select.items()

list_products = []
for k, v in count_keywords.items():
    list_products.append([keywords_select[k], v])
list_products.sort(key=lambda x: x[1], reverse=True)

liste = sorted(list_products, key=lambda x: x[1], reverse=True)

liste

plt.rc('font', family='AppleGothic')
plt.rc('font', weight='normal')
fig, ax = plt.subplots(figsize=(10, 30))
y_axis = [i[1] for i in liste[:125]]
x_axis = [k for k, i in enumerate(liste[:125])]
x_label = [i[0] for i in liste[:125]]
plt.xticks(fontsize=15)
plt.yticks(fontsize=13)
plt.yticks(x_axis, x_label)
plt.xlabel("발생 횟수", fontsize=18, labelpad=10)
ax.barh(x_axis, y_axis, align='center')
ax = plt.gca()
ax.invert_yaxis()

plt.title("단어 발생 빈도 그래프", bbox={'facecolor': 'k', 'pad': 5}, color='w', fontsize=25)
plt.show()

count_keywords

# list_products = []

# # Loop through the count_keywords and check the different conditions
# for k,v in count_keywords.items():
#     word = keywords_select[k]
#     if word in ['pink', 'blue', 'tag', 'green', 'orange']:
#         continue
#     if len(word) < 3 or v < 13:
#         continue
#     if ('+' in word) or ('/' in word):
#         continue
#     list_products.append([word, v])

# # list most kept words
# list_products.sort(key = lambda x:x[1], reverse = True)
# print('words kept:', len(list_products))

liste_products = df_cleaned['Description'].unique()
X = pd.DataFrame()
for key, occurence in list_products:
    X.loc[:, key] = list(map(lambda x: int(key.upper() in x), liste_products))

threshold = [0, 1, 2, 3, 5, 10]
label_col = []

for i in range(len(threshold)):
    if i == len(threshold) - 1:
        col = '.>{}'.format(threshold[i])
    else:
        col = '{}<.<{}'.format(threshold[i], threshold[i + 1])
    label_col.append(col)
    X.loc[:, col] = 0

for i, prod in enumerate(liste_products):
    prix = df_cleaned[df_cleaned['Description'] == prod]['UnitPrice'].mean()
    j = 0
    while prix > threshold[j]:
        j += 1
        if j == len(threshold): break
    X.loc[i, label_col[j - 1]] = 1

"""### 2. K-Means Clustering 과 PCA

#### 1) Silhouette Score 기반 K
"""

X.head()

for n_clusters in range(3, 10):
    kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    print("클러스터 개수 =", n_clusters, "silhouette 평균 점수 :", silhouette_avg)

n_clusters = 5

kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=30)
kmeans.fit(X)
clusters = kmeans.predict(X)

pd.Series(clusters).value_counts()

"""#### 2) PCA"""
pca = PCA()
pca.fit(X)
pca_samples = pca.transform(X)

fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)

plt.rc('font', family='AppleGothic')
plt.step(range(X.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='누적된 분산력')

sns.barplot(x=np.arange(1, X.shape[1] + 1), y=pca.explained_variance_ratio_, alpha=0.5, color='g',
            label='개별적인 분산력')

plt.xlim(0, 100)

ax.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('설명 가능한 분산력', fontsize=14)
plt.xlabel('주성분 개수', fontsize=14)
plt.legend(loc='upper left', fontsize=13);
plt.show()

pca = PCA(n_components=50)
matrix_9D = pca.fit_transform(X)
mat = pd.DataFrame(matrix_9D)
mat['cluster'] = pd.Series(clusters)

import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0: 'r', 1: 'gold', 2: 'b', 3: 'k', 4: 'c', 5: 'g'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize=(15, 8))
increment = 0
for ix in range(4):
    for iy in range(ix + 1, 4):
        increment += 1
        ax = fig.add_subplot(2, 3, increment)
        ax.scatter(mat[ix], mat[iy], c=label_color, alpha=0.4)
        plt.ylabel('PCA {}'.format(iy + 1), fontsize=12)
        plt.xlabel('PCA {}'.format(ix + 1), fontsize=12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if increment == 9: break
    if increment == 9: break

comp_handler = []
for i in range(5):
    comp_handler.append(mpatches.Patch(color=LABEL_COLOR_MAP[i], label=i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.97),
           title='Cluster', facecolor='lightgrey',
           shadow=True, frameon=True, framealpha=1,
           fontsize=13, bbox_transform=plt.gcf().transFigure)

plt.show()

corresp = dict()
for key, val in zip(liste_products, clusters):
    corresp[key] = val

df_cleaned['categ_product'] = df_cleaned.loc[:, 'Description'].map(corresp)

for i in range(5):
    col = 'categ_{}'.format(i)
    df_temp = df_cleaned[df_cleaned['categ_product'] == i]
    price_temp = df_temp['UnitPrice'] * (df_temp['Quantity'] - df_temp['QuantityCanceled'])
    price_temp = price_temp.apply(lambda x: x if x > 0 else 0)
    df_cleaned.loc[:, col] = price_temp
    df_cleaned[col].fillna(0, inplace=True)

df_cleaned[['InvoiceNo', 'Description', 'categ_product', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']][:5]

temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['TotalPrice'].sum()
basket_price = temp.rename(columns={'TotalPrice': 'Basket Price'})
basket_price

for i in range(5):
    col = 'categ_{}'.format(i)
    temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)[col].sum()
    basket_price.loc[:, [col]] = temp

df_cleaned['InvoiceDate_int'] = df_cleaned['InvoiceDate'].astype('int64')
temp = df_cleaned.groupby(by=['CustomerID', 'InvoiceNo'], as_index=False)['InvoiceDate_int'].mean()
df_cleaned.drop('InvoiceDate_int', axis=1, inplace=True)
basket_price.loc[:, 'InvoiceDate'] = pd.to_datetime(temp['InvoiceDate_int'])

basket_price = basket_price[basket_price['Basket Price'] > 0]
basket_price.sort_values('CustomerID', ascending=True)[:5]

print(basket_price['InvoiceDate'].min(), '->', basket_price['InvoiceDate'].max())

# number of visits and stats on the basket amount / users
transactions_per_user = basket_price.groupby(by=['CustomerID'])['Basket Price'].agg(
    ['count', 'min', 'max', 'mean', 'sum'])
for i in range(5):
    col = 'categ_{}'.format(i)
    transactions_per_user.loc[:, col] = basket_price.groupby(by=['CustomerID'])[col].sum() / \
                                        transactions_per_user['sum'] * 100

transactions_per_user.reset_index(drop=False, inplace=True)
basket_price.groupby(by=['CustomerID'])['categ_0'].sum()
transactions_per_user.sort_values('CustomerID', ascending=True)[:5]

last_date = basket_price['InvoiceDate'].max().date()

first_registration = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].min())
last_purchase = pd.DataFrame(basket_price.groupby(by=['CustomerID'])['InvoiceDate'].max())

test = first_registration.applymap(lambda x: (last_date - x.date()).days)
test2 = last_purchase.applymap(lambda x: (last_date - x.date()).days)

transactions_per_user.loc[:, 'LastPurchase'] = test2.reset_index(drop=False)['InvoiceDate']
transactions_per_user.loc[:, 'FirstPurchase'] = test.reset_index(drop=False)['InvoiceDate']

transactions_per_user[:5]

n1_check = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
n1_check

n2_check = transactions_per_user.shape[0]
n2_check

n1 = transactions_per_user[transactions_per_user['count'] == 1].shape[0]
n2 = transactions_per_user.shape[0]
print("no. of customers with single purchase: {:<2}/{:<5} ({:<2.2f}%)".format(n1, n2, n1 / n2 * 100))

list_cols = ['count', 'min', 'max', 'mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']

#
selected_customers = transactions_per_user.copy(deep=True)
X_selected = selected_customers[list_cols]

X_selected

scaler = StandardScaler()
scaler.fit(X_selected)
print('variables mean values: \n' + 90 * '-' + '\n', scaler.mean_)
scaled_X_selected = scaler.transform(X_selected)

scaled_X_selected

pca = PCA()
pca.fit(scaled_X_selected)
pca_samples = pca.transform(scaled_X_selected)

fig, ax = plt.subplots(figsize=(14, 5))
sns.set(font_scale=1)
plt.step(range(X_selected.shape[1]), pca.explained_variance_ratio_.cumsum(), where='mid',
         label='cumulative explained variance')
sns.barplot(x=np.arange(1, X_selected.shape[1] + 1), y=pca.explained_variance_ratio_, alpha=0.5, color='g',
            label='individual explained variance')
plt.xlim(0, 10)

ax.set_xticklabels([s if int(s.get_text()) % 2 == 0 else '' for s in ax.get_xticklabels()])

plt.ylabel('Explained variance', fontsize=14)
plt.xlabel('Principal components', fontsize=14)
plt.legend(loc='best', fontsize=13)
plt.show()

n_clusters = 11
kmeans = KMeans(init='k-means++', n_clusters=n_clusters, n_init=100)
kmeans.fit(scaled_X_selected)
clusters_clients = kmeans.predict(scaled_X_selected)
silhouette_avg = silhouette_score(scaled_X_selected, clusters_clients)
print('silhouette score: {:<.3f}'.format(silhouette_avg))

pd.DataFrame(pd.Series(clusters_clients).value_counts(), columns=['no. of customers']).T

pca = PCA(n_components=6)
X_selected_3D = pca.fit_transform(scaled_X_selected)
mat = pd.DataFrame(X_selected_3D)
mat['cluster'] = pd.Series(clusters_clients)

mat.head()

"""#### 3) 시각화"""

import matplotlib.patches as mpatches

sns.set_style("white")
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2.5})

LABEL_COLOR_MAP = {0: 'r', 1: 'tan', 2: 'b', 3: 'k', 4: 'c', 5: 'g', 6: 'deeppink', 7: 'skyblue', 8: 'darkcyan',
                   9: 'orange',
                   10: 'yellow', 11: 'tomato', 12: 'seagreen'}
label_color = [LABEL_COLOR_MAP[l] for l in mat['cluster']]

fig = plt.figure(figsize=(12, 10))
increment = 0
for ix in range(6):
    for iy in range(ix + 1, 6):
        increment += 1
        ax = fig.add_subplot(4, 3, increment)
        ax.scatter(mat[ix], mat[iy], c=label_color, alpha=0.5)
        plt.ylabel('PCA {}'.format(iy + 1), fontsize=12)
        plt.xlabel('PCA {}'.format(ix + 1), fontsize=12)
        ax.yaxis.grid(color='lightgray', linestyle=':')
        ax.xaxis.grid(color='lightgray', linestyle=':')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        if increment == 12: break
    if increment == 12: break

comp_handler = []
for i in range(n_clusters):
    comp_handler.append(mpatches.Patch(color=LABEL_COLOR_MAP[i], label=i))

plt.legend(handles=comp_handler, bbox_to_anchor=(1.1, 0.9),
           title='Cluster', facecolor='lightgrey',
           shadow=True, frameon=True, framealpha=1,
           fontsize=13, bbox_transform=plt.gcf().transFigure)

plt.tight_layout()

selected_customers.loc[:, 'cluster'] = clusters_clients

"""### 3. 고객군 분류 예측 모델링"""


class Class_Fit(object):
    def __init__(self, clf, params=None):
        if params:
            self.clf = clf(**params)
        else:
            self.clf = clf()

    # train on the x_train and y_train data
    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    # predict/validate on the held-out data
    def predict(self, x):
        return self.clf.predict(x)

    # hyperparameter tuning using GridsearchCV
    def grid_search(self, parameters, Kfold):
        self.grid = GridSearchCV(estimator=self.clf, param_grid=parameters, cv=Kfold)

    # fit after tuning
    def grid_fit(self, X, Y):
        self.grid.fit(X, Y)

    # predict
    def grid_predict(self, X, Y):
        self.predictions = self.grid.predict(X)
        print("Precision: {:.2f} % ".format(100 * metrics.accuracy_score(Y, self.predictions)))


columns = ['mean', 'categ_0', 'categ_1', 'categ_2', 'categ_3', 'categ_4']
X = selected_customers[columns]
Y = selected_customers['cluster']

Y.value_counts()

X.head()

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size=0.8, random_state=0)

X_train.shape, Y_train.shape

X_test.shape, Y_test.shape

svc = Class_Fit(clf=svm.LinearSVC)
svc.grid_search(parameters=[{'C': np.logspace(-2, 2, 10)}], Kfold=5)

svc.grid_fit(X=X_train, Y=Y_train)

svc.grid_predict(X_test, Y_test)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


class_names = [i for i in range(11)]

cnf_matrix = confusion_matrix(Y_test, svc.predictions)

np.set_printoptions(precision=2)

plt.figure(figsize=(8, 8))

plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=False, title='Confusion matrix')

lr = Class_Fit(clf=linear_model.LogisticRegression)
lr.grid_search(parameters=[{'C': np.logspace(-2, 2, 20)}], Kfold=5)

# fit on training data
lr.grid_fit(X=X_train, Y=Y_train)

# predict on test data
lr.grid_predict(X_test, Y_test)

"""#### k-Nearest Neighbors"""

knn = Class_Fit(clf=neighbors.KNeighborsClassifier)
knn.grid_search(parameters=[{'n_neighbors': np.arange(1, 50, 1)}], Kfold=5)

# fit on training data
knn.grid_fit(X=X_train, Y=Y_train)

# predict on test data
knn.grid_predict(X_test, Y_test)

"""#### Decision Tree"""

tr = Class_Fit(clf=tree.DecisionTreeClassifier)
tr.grid_search(parameters=[{'criterion': ['entropy', 'gini'], 'max_features': ['sqrt', 'log2']}], Kfold=5)

# fit on training data
tr.grid_fit(X=X_train, Y=Y_train)

# predict on test data
tr.grid_predict(X_test, Y_test)

"""#### Random Forest"""

rf = Class_Fit(clf=ensemble.RandomForestClassifier)
param_grid = {'criterion': ['entropy', 'gini'], 'n_estimators': [20, 40, 60, 80, 100],
              'max_features': ['sqrt', 'log2']}

rf.grid_search(parameters=param_grid, Kfold=5)

# fit on training data
rf.grid_fit(X=X_train, Y=Y_train)

# predict on test data
rf.grid_predict(X_test, Y_test)

"""#### AdaBoost Classifier"""

ada = Class_Fit(clf=AdaBoostClassifier)
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
ada.grid_search(parameters=param_grid, Kfold=5)

# fit on training data
ada.grid_fit(X=X_train, Y=Y_train)

# predict on test data
ada.grid_predict(X_test, Y_test)

"""#### Gradient Boosting Classifier"""

gb = Class_Fit(clf=ensemble.GradientBoostingClassifier)
param_grid = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
gb.grid_search(parameters=param_grid, Kfold=5)

# fit on training data
gb.grid_fit(X=X_train, Y=Y_train)

# predict on test data
gb.grid_predict(X_test, Y_test)

"""#### Voting Classifier"""

# random forest classifier best params
rf_best = ensemble.RandomForestClassifier(**rf.grid.best_params_)

# gradient boosting classifier best params
gb_best = ensemble.GradientBoostingClassifier(**gb.grid.best_params_)

# support vector classifier best params
svc_best = svm.LinearSVC(**svc.grid.best_params_)

# decision tree classifier best params
tr_best = tree.DecisionTreeClassifier(**tr.grid.best_params_)

# k-nearest neighbor classifier best params
knn_best = neighbors.KNeighborsClassifier(**knn.grid.best_params_)

# logistics regression best params
lr_best = linear_model.LogisticRegression(**lr.grid.best_params_)

votingC = ensemble.VotingClassifier(estimators=[('rf', rf_best), ('gb', gb_best), ('lr', lr_best)], voting='soft')
votingC = votingC.fit(X_train, Y_train)
predictions = votingC.predict(X_test)

print("Precision: {:.2f} % ".format(100 * metrics.accuracy_score(Y_test, predictions)))

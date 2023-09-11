# -*- coding: utf-8 -*-
"""[전체]Ch14. [실습12] 온라인 쇼핑몰 고객 Clustering 분석.py

* [olist dataset] : https://www.kaggle.com/olistbr/brazilian-ecommerce

#Part4. [실습12] 온라인 쇼핑몰 고객 Clustering 분석

## 01. 데이터 소개 및 분석프로세스 수립
 : "강의자료 → Ch04. [실습12] 온라인 쇼핑몰 고객 Clustering 분석" 참고

"""

from matplotlib import pyplot as plt

plt.rc('font', family='AppleGothic')
############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter

import time
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer

from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from sklearn.decomposition import PCA

### 0. 데이터 불러오기
df_orders = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_orders_dataset.csv', sep=',',
                        parse_dates=['order_purchase_timestamp', 'order_estimated_delivery_date',
                                     'order_delivered_customer_date'])
df_customers = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_customers_dataset.csv',
                           sep=',')
df_payments = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_order_payments_dataset.csv',
                          sep=',')
df_reviews = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_order_reviews_dataset.csv',
                         sep=',')
df_geolocation = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_geolocation_dataset.csv",
                             sep=',')
df_items = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_order_items_dataset.csv', sep=',')
df_products = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_products_dataset.csv', sep=',')
product_translations = pd.read_csv(
    '/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/product_category_name_translation.csv', sep=',')
df_sellers = pd.read_csv('/Users/jiyoonkim/Documents/da_study/comFiles/ECommerce/olist_sellers_dataset.csv', sep=',')

"""### 1. 데이터 전처리"""

print(df_customers.columns)

"""> customer_unique_id"""

df_customers.groupby('customer_unique_id').size().value_counts()

"""#### 고객 기준 데이터 병합

> df_products 와 product_translations 합치기
"""

df_products.head()
product_translations.head()
df_products = df_products.merge(product_translations, how='left', left_on='product_category_name',
                                right_on='product_category_name')
print(df_products.columns)

"""> df_orders 와 df_customers 합치기"""

df_orders.head()
df_customers.head()
df_order_by_customer = pd.merge(df_orders[['customer_id', 'order_id', 'order_purchase_timestamp']],
                                df_customers[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix']],
                                how='inner', on='customer_id')
df_order_by_customer = df_order_by_customer.drop_duplicates(keep=False)
print(df_order_by_customer.columns)
print(df_order_by_customer.shape)
print(df_order_by_customer["customer_id"].duplicated().value_counts())

"""> df_order_by_customer 와 df_payments 합치기 : df_payment_and_order"""
df_order_by_customer.head()
df_payments.head()
df_payment_and_order = df_order_by_customer.merge(df_payments, how='inner', left_on='order_id', right_on='order_id')
df_payment_and_order.head()

"""> df_payment_and_order 를 customer_unique_id, customer_id, order_id, ord_purchase_timestamp 를 기준으로 합을 구하기"""
df_payment_by_customer = df_payment_and_order.groupby(
    ['customer_unique_id', 'customer_id', 'order_id', 'order_purchase_timestamp']).sum().reset_index()
df_payment_by_customer.head()

print(df_payment_by_customer.columns)
print(df_payment_by_customer.shape)
print(df_payment_by_customer["customer_id"].duplicated().value_counts())

"""> df_payment_by_customer 와 df_items 합치기 : df_items_paid"""

df_items.head()

print(max(df_items['order_item_id']))

df_items_paid = df_payment_by_customer.merge(
    df_items[['order_id', 'order_item_id', 'product_id', 'price', 'freight_value']], how='inner', left_on='order_id',
    right_on='order_id')
print(df_items_paid.shape)
print(df_items_paid["customer_id"].duplicated().value_counts())

"""> df_items_paid 와 df_products 합치기 : df_products_bought"""

df_products_bought = df_items_paid.merge(df_products[['product_id', 'product_category_name_english']], how='inner',
                                         left_on='product_id', right_on='product_id')

print(df_products_bought.shape)
print(df_products_bought.columns)
print(df_products_bought["customer_id"].duplicated().value_counts())
# insert a columns for the amount of products categories for each order

"""> df_products_bought 와 df_reviews 합치기 : df_reviews_by_customer"""

df_reviews_by_customer = df_products_bought.merge(df_reviews[['order_id', 'review_score']], how='inner',
                                                  left_on='order_id', right_on='order_id')
print(df_reviews_by_customer.shape)
print(df_reviews_by_customer.columns)
print(df_reviews_by_customer["customer_id"].duplicated().value_counts())

"""> df_reviews_by_customer 를 customer_unique_id 와 customer_id 기준으로 aggregate"""

gb_products_bought = df_reviews_by_customer.groupby(['customer_unique_id', 'customer_id']).agg({
    'order_item_id': 'sum',
    'product_category_name_english': 'count',
    'price': 'sum',
    'freight_value': 'sum',
    'payment_value': 'sum',
    'review_score': 'mean',
    'order_purchase_timestamp': 'max',
    'order_id': 'count'}).reset_index()

alldata = gb_products_bought.sort_values(by=['product_category_name_english'], ascending=False)

alldata = alldata.rename(columns={"product_category_name_english": "amount_prod_categories"})

print(alldata.shape)
print(alldata.columns)
print(alldata["customer_id"].duplicated().value_counts())

alldata.head()

"""#### 새로운 컬럼 만들기

> 가장 최근 구매일로부터 며칠 전인가 : recency
"""

latestdate = np.max(alldata['order_purchase_timestamp'])
print(latestdate)

alldata['recency'] = alldata['order_purchase_timestamp'].apply(lambda x: (latestdate - x).days)

alldata['recency']

alldata = alldata.drop(['order_purchase_timestamp'], axis=1)

alldata = alldata.drop(['customer_unique_id'], axis=1)
alldata.head()

"""> 총 몇 번 구매하였는가 : order_id -> frequency"""

alldata = alldata.rename(columns={"order_id": "frequency"})
alldata.head()

alldata.describe()

"""#### 판매자 기준 데이터 병합"""

df_items.head()

df_sellers.head()

"""> df_items 와 df_sellers 를 seller_id 기준으로 병합 -> df_items_sold"""

df_items_sold = df_items.merge(df_sellers[['seller_id', 'seller_zip_code_prefix']], how='inner', left_on='seller_id',
                               right_on='seller_id')
df_items_sold.columns

"""> df_items_sold 와 df_reviews 를 order_id 기준으로 병합 -> df_items_reviewed"""

df_items_reviewed = df_items_sold.merge(df_reviews[['order_id', 'review_score']], how='inner', left_on='order_id',
                                        right_on='order_id')

"""> df_items_reviewed 와 df_payments 를 order_id 기준으로 병합 -> df_sales"""

df_sales = df_items_reviewed.merge(df_payments[['order_id', 'payment_value']], how='inner', left_on='order_id',
                                   right_on='order_id')
df_sales.columns

"""> 판매자별 총 판매액과 평균 평점 구하기"""

gb_seller = df_sales.groupby(['seller_id']).agg({'payment_value': 'sum', 'review_score': 'mean'}).reset_index()
gb_seller = gb_seller.rename(columns={"payment_value": "total_revenue"})
print(gb_seller.shape)
print(gb_seller["seller_id"].duplicated().value_counts())

gb_seller.head()

"""### 2. 데이터 정제

#### 중복값
"""

# 중복된 항목 수 알아보기
print("중복된 항목 수 :", len(alldata[alldata.duplicated()]))

"""#### 결측값"""

missing_df = alldata.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df

"""#### 각 컬럼별 유일한 값의 비율"""

i = 0
colnum = []
colname = []
nullpc = []
uniq = []

numrows = len(alldata)

for col in alldata:
    i += 1

    uniques = ((len(alldata[col].unique().tolist())) / numrows) * 100
    colnum.append(i)
    colname.append(col)
    uniq.append(uniques)

df_nul = pd.DataFrame({'num': colnum, 'name': colname, '% unique values': uniq})
df_nul = df_nul.sort_values(by='% unique values', ascending=False)
df_nul

"""#### 이상치"""

plt.figure(figsize=(15, 4))
sns.boxplot(data=alldata, orient="h")
plt.show()
print(alldata.columns)

alldata.shape

alldata1 = alldata[alldata['payment_value'] > 6000]
print(alldata1.shape)

alldata2 = alldata[alldata['payment_value'] < 6000]
plt.figure(figsize=(15, 4))
sns.boxplot(data=alldata2, orient="h")
plt.show()

print(alldata2.shape)

"""> 이상치 제거"""


def detect_outliers(df, features):
    outlier_indices = []

    for col in features:
        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)
        # 3rd quartile (75%)
        Q3 = np.percentile(df[col], 75)
        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 8 * IQR

        # 이상치 컬럼 리스트
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)

    outlier_indices = Counter(outlier_indices)
    return outlier_indices


lof = ['price', 'payment_value']
Outliers_to_drop = detect_outliers(alldata, lof)

print(len(Outliers_to_drop))

alldata = alldata.drop(Outliers_to_drop)

print(alldata.shape)

"""### 3. 데이터 탐색

#### 판매자 데이터
"""

gb_seller.head()

best_noted_sellers = gb_seller[:1000]
best_sellers = best_noted_sellers[['seller_id', 'review_score', 'total_revenue']]

"""> 판매자 평점과 매출 관계"""

sellerpop = plt.scatter(y=best_sellers['review_score'], x=best_sellers["total_revenue"], marker='+', color='tomato');
sns.set_context("talk")
plt.xlabel('Revenue')
plt.ylabel('Review average note')
plt.title('Seller rating and revenue')
plt.show()

customer_satisfaction = alldata.sort_values(by=['review_score'], ascending=False)

satisfactiongraph = plt.hist(customer_satisfaction['review_score'])
plt.xlabel('판매자 평점')
plt.ylabel('누적 횟수')

satisfactiongraph = sns.displot(gb_seller['review_score'], kde=True)
plt.xlabel('판매자 평점')
plt.ylabel('누적 횟수')
plt.show()

"""#### 제품 구매 선호도

> WordCloud
"""

missing_df = df_reviews_by_customer.isnull().sum(axis=0).reset_index()
missing_df.columns = ['column_name', 'missing_count']
missing_df = missing_df.loc[missing_df['missing_count'] > 0]
missing_df = missing_df.sort_values(by='missing_count')
missing_df

df_reviews_by_customer['product_category_name_english'].fillna("None", inplace=True)

soup = ' '.join(df_reviews_by_customer['product_category_name_english'])

wordcloud = WordCloud(width=1000, height=500, max_words=50)
wordcloud.generate(soup)
plt.figure(figsize=(20, 10), facecolor='k')
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# product_category_name_english 별 평균 구하기
product_rew = df_reviews_by_customer.groupby(['product_category_name_english']).mean(numeric_only=True).reset_index()
# 평점으로 정렬하기
product_rew = product_rew.sort_values(by=['review_score'], ascending=False)

best_noted_products = product_rew.head(10)

best_noted_products

prodnote_hist = sns.barplot(y=best_noted_products["product_category_name_english"],
                            x=best_noted_products["review_score"]);
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize': (10, 8)})
plt.xlabel('평균 평점', fontsize=14)
plt.ylabel('제품 분류', fontsize=14)
plt.show()

# product_category_name_english 별 합계 구하기
prodcat = df_reviews_by_customer.groupby(['product_category_name_english']).sum(numeric_only=True).reset_index()
prodcat = prodcat[prodcat['product_category_name_english'] != 'None']
# order_item_id 기준으로 정렬하기
prodcat = prodcat.sort_values(by=['order_item_id'], ascending=False)
prodcat.head()

prodcat0 = prodcat[:10]
catorderhist = sns.barplot(y=prodcat0["product_category_name_english"], x=prodcat0["order_item_id"]);
sns.set_context("talk")
sns.set_style("whitegrid")
sns.set(rc={'figure.figsize': (10, 8)})
plt.xlabel('제품 분류', fontsize=14)
plt.ylabel('총 주문 수', fontsize=14)
plt.show()

prodcat = prodcat.sort_values(by=['payment_value'], ascending=False)
prodcat1 = prodcat[:10]

cat_revenue_hist = sns.barplot(y=prodcat1["product_category_name_english"], x=prodcat1["payment_value"]);
sns.set(rc={'figure.figsize': (10, 15)})
plt.xlabel('총 매출액', fontsize=14)
plt.ylabel('제품 분류', fontsize=14)
plt.show()

plt.figure(figsize=(10, 5))
plt.title('각 주별 고객 정보')
plt.ylabel('고객 수')
plt.xlabel('각 주 이름')
sns.barplot(x=df_customers['customer_state'].value_counts().index,
            y=df_customers['customer_state'].value_counts().values)
plt.show()
"""## 3. RFM 분석하기

* Recency: 고객이 최근에 구입을 했는가?
* Frequency: 고객이 얼마나 빈번하게 상품을 구입했는가?
* Monetary: 고객이 구입했던 총 금액은 얼마인가?
"""

transaction_data = alldata[['customer_id', 'frequency', 'recency', 'payment_value']]
print(transaction_data.shape)
transaction_data.head()

df_RFM = transaction_data.rename(columns={"payment_value": "monetary_value"})
df_RFM = df_RFM.reset_index()

df_RFM.head()

"""
    frequency: # of days in which a customer made a repeat purchase
    T: customer's age in days
    recency: customer's age in days at time of most recent purchase
    monetary_value: sum of a customer's purchases

"""

df_RFM = df_RFM[df_RFM['frequency'] > 0]
df_RFM.describe()

quintiles = df_RFM[['recency', 'frequency', 'monetary_value']].quantile([.2, .4, .6, .8]).to_dict()
quintiles


def r_score(x):
    if x <= quintiles['recency'][.2]:
        return 5
    elif x <= quintiles['recency'][.4]:
        return 4
    elif x <= quintiles['recency'][.6]:
        return 3
    elif x <= quintiles['recency'][.8]:
        return 2
    else:
        return 1


def fm_score(x, c):
    if x <= quintiles[c][.2]:
        return 1
    elif x <= quintiles[c][.4]:
        return 2
    elif x <= quintiles[c][.6]:
        return 3
    elif x <= quintiles[c][.8]:
        return 4
    else:
        return 5


df_RFM['R'] = df_RFM['recency'].apply(lambda x: r_score(x))
df_RFM['F'] = df_RFM['frequency'].apply(lambda x: fm_score(x, 'frequency'))
df_RFM['M'] = df_RFM['monetary_value'].apply(lambda x: fm_score(x, 'monetary_value'))

df_RFM['RFM Score'] = df_RFM['R'].map(str) + df_RFM['F'].map(str) + df_RFM['M'].map(str)
df_RFM.head()

segt_map = {
    r'[1-2][1-2]': '최근구매없음/구매횟수적음',
    r'[1-2][3-4]': '최근구매없음/구매횟수보통',
    r'[1-2]5': '최근구매없음/구매횟수많음',
    r'3[1-2]': '비교적최근구매함/구매횟수적음',
    r'33': '비교적최근구매함/구매횟수보통',
    r'[3-4][4-5]': '최근구매함/구매횟수많음',
    r'41': '최근구매함/구매횟수적음',
    r'51': '방금전구매함/구매횟수적음',
    r'[4-5][2-3]': '방금전구매함/구매횟수보통/프로모션대상',
    r'5[4-5]': '방금전구매함/구매횟수많음'
}

df_RFM['Segment'] = df_RFM['R'].map(str) + df_RFM['F'].map(str)
df_RFM['Segment'] = df_RFM['Segment'].replace(segt_map, regex=True)
df_RFM.head()

segments_counts = df_RFM['Segment'].value_counts().sort_values(ascending=True)

fig, ax = plt.subplots()

bars = ax.barh(range(len(segments_counts)),
               segments_counts,
               color='b')

ax.set_frame_on(False)
ax.tick_params(left=False,
               bottom=False,
               labelbottom=False)

ax.set_yticks(range(len(segments_counts)))
ax.set_yticklabels(segments_counts.index, fontsize=14)

for i, bar in enumerate(bars):
    value = bar.get_width()
    if segments_counts.index[i] in ['방금전구매함/구매횟수보통/프로모션대상', '방금전구매함/구매횟수많음']:
        bar.set_color('r')
    ax.text(value,
            bar.get_y() + bar.get_height() / 2,
            '{:,} ({:}%)'.format(int(value),
                                 int(value * 100 / segments_counts.sum())),
            va='center',
            ha='left'
            )
sns.set(rc={'figure.figsize': (5, 10)})
plt.show()

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale

"""## 4. Unsupervised Segmentation

### K-Means Clustering
"""

alldata.columns

"""#### 1) Feature Selection"""

sns.set(style="white")
corrmat = alldata.corr(numeric_only=True)

f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);
plt.show()

num_feat = alldata.select_dtypes('number').columns.values
comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(alldata[comb[0]], alldata[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)

high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.95]
high_corr_num

grouping_variables = ['amount_prod_categories', 'payment_value', 'freight_value', 'review_score', 'recency']

df_reduced = alldata[grouping_variables]

plt.figure(figsize=(15, 4))
sns.boxplot(data=df_reduced, orient="h")
plt.show()

"""#### 2) Scailing"""

X = df_reduced.values
X_scaled = scale(X)
print('X meanX', np.mean(X_scaled), ',X standard deviation:', np.std(X_scaled))

"""> 결측값을 평균값으로 Imputation"""

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(X_scaled)
X_scaled = imputer.transform(X_scaled)

scaled_dataframe = pd.DataFrame(X_scaled, columns=df_reduced.columns)
plt.figure(figsize=(15, 4))
sns.boxplot(data=scaled_dataframe, orient="h")
plt.show()

"""#### 3) 데이터 준비"""

train_size = int(len(X) * 0.98)
X_train, X_test = X[0:train_size], X[train_size:len(X)]

print('Observations: %d' % (len(X)))
print('Training Observations: %d' % (len(X_train)))
print('Testing Observations: %d' % (len(X_test)))

"""#### 4) Hyperparameters Optimization

##### Elbow Method
"""

from sklearn.cluster import KMeans

wcss_all = []
allgaps = []


def optimalK(data, nrefs=3, maxClusters=10):
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        wcss = np.zeros(nrefs)

        # 랜덤 샘플링으로 형태가 같은 데이터를 만들고 주어진 n에 대하여 클러스터링
        for i in range(nrefs):
            randomReference = np.random.random_sample(size=data.shape)
            kmeans = KMeans(k)
            kmeans.fit(randomReference)

            w = kmeans.inertia_
            wcss[i] = w

        # 분석 대상 데이터로 주어진 n에 대하여 클러스터링
        kmeans = KMeans(n_clusters=k, init="k-means++", random_state=10)
        kmeans.fit(data)

        origin_w = kmeans.inertia_

        # 랜덤 샘플링 데이터에 대한 wcss 의 평균과 분석 대상 데이터의 Inertia value 의 차이를 gap 으로 저장
        gap = np.log(np.mean(wcss)) - np.log(origin_w)
        gaps[gap_index] = gap
        df_merge = pd.DataFrame({'clusterCount': k, 'gap': gap, 'origin_w': round(origin_w / 100000000, 1)}, index=[0])
        resultsdf = pd.concat([resultsdf, df_merge], ignore_index=True)
        allgaps.append(gap)
        wcss_all.append(origin_w)

    print(resultsdf)
    return (resultsdf)


resultsdf = optimalK(X_train, nrefs=3, maxClusters=10)

plt.figure(figsize=(5, 5))
plt.plot(range(1, 10), allgaps)
plt.grid(True)
plt.xlabel('Cluster Count')
plt.ylabel('Gap Value')
plt.title('Gap Values by Cluster Count')
plt.show()

plt.figure(figsize=(5, 5))
plt.plot(range(1, 10), wcss_all)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WSS')
plt.show()

"""k=4"""

kmeans = KMeans(n_clusters=4, init="random", random_state=10)
clusts_train = kmeans.fit_predict(X_train)

clusts_test = np.zeros(len(X_test))
clusts_test = clusts_test + 42
clusts_test

clust = np.concatenate((clusts_train, clusts_test), axis=None)
print(clusts_train.shape, clusts_test.shape, clust.shape)

df_train = pd.DataFrame.from_records(X_train)
df_test = pd.DataFrame.from_records(X_test)

df_clustered = pd.concat([df_train, df_test])
df_clustered.head()

"""#### 5) 시각화

##### PCA
"""

pca_feat = PCA(n_components=2)
X_PCA = pca_feat.fit_transform(df_clustered)
print(pca_feat.explained_variance_ratio_)

features = range(pca_feat.n_components_)

df_PCA = pd.DataFrame.from_records(X_PCA)
clust_series = pd.Series(clust)
clust_series.head()

df_PCA = df_PCA.assign(cluster=clust_series)
df_PCA.columns

df_PCA = df_PCA.rename(columns={0: 'PCA_1', 1: 'PCA_2'})
df_PCA["cluster"].replace({0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 42: "Test values"},
                          inplace=True)
df_PCA.head()

df_PCA = df_PCA.dropna()
print(df_PCA.columns, df_PCA.shape)

plt.rc('font', family='AppleGothic')
orderhue = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Test values']
plt.rc('font', family='AppleGothic')
sns.lmplot(x='PCA_1', y='PCA_2', data=df_PCA, fit_reg=False, legend=False, hue="cluster", hue_order=orderhue,
           scatter_kws={"s": 200, "alpha": 0.3})
plt.xlabel('PCA component 1', fontsize=14)
plt.ylabel('PCA component 2', fontsize=14)
plt.title('PCA 를 이용한 K-Means Cluster Plot', fontsize=20)
plt.legend(fontsize=14)
plt.show()

"""##### T-SNE"""

tsne = TSNE(n_components=2, perplexity=31, random_state=2020)

X_concat = df_clustered.values

X_TSNE = tsne.fit_transform(X_concat)

df_TSNE = pd.DataFrame.from_records(X_TSNE)
print(df_TSNE.shape)
df_TSNE.head()

clust_series = pd.Series(clust)

df_TSNE = df_TSNE.assign(cluster=clust_series)

print(df_TSNE.columns, df_TSNE.shape)
df_TSNE = df_TSNE.rename(columns={0: 'TSNE_1', 1: 'TSNE_2'})
df_TSNE["cluster"].replace({0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 42: "Test values"},
                           inplace=True)
df_TSNE.head()

df_TSNE["cluster"].value_counts()

orderhue = ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4', 'Test values']
plt.rc('font', family='AppleGothic')
sns.lmplot(x='TSNE_1', y='TSNE_2', data=df_TSNE, fit_reg=False, legend=True, hue="cluster", hue_order=orderhue,
           scatter_kws={"s": 200, "alpha": 0.15})
plt.xlabel('TSNE component 1', fontsize=14)
plt.ylabel('TSNE component 2', fontsize=14)
plt.title('T-SNE 를 이용한 K-Means Cluster Plot', fontsize=20)
plt.show()

"""##### BoxPlot"""

df_clustered_labeled = df_clustered.assign(cluster=clust_series)
df_clustered_labeled.head()

df_clustered_labeled.cluster.unique()

columnsvector = df_reduced.columns
print(columnsvector)
df_train_labeled = df_train
df_train_labeled.columns = columnsvector

df_train_labeled = df_train_labeled.assign(cluster=clusts_train)
df_train_labeled["cluster"].replace({0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4"}, inplace=True)
df_train_labeled.head()

x1 = df_train_labeled["cluster"]
y0 = df_train_labeled[df_train_labeled.columns[0]]
y1 = df_train_labeled[df_train_labeled.columns[1]]
y2 = df_train_labeled[df_train_labeled.columns[2]]
y3 = df_train_labeled[df_train_labeled.columns[3]]
y4 = df_train_labeled[df_train_labeled.columns[4]]

order = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]

plt.figure(figsize=(16, 16))

plt.xlabel('')
plt.subplot(321)
plt.xticks([])
df_train_labeled

sns.boxplot(x=x1, y=y0, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')
plt.subplot(322)
plt.xticks([])

sns.boxplot(x=x1, y=y1, order=order, showfliers=True, data=df_train_labeled)
plt.xlabel('')

plt.subplot(323)
plt.xticks([])

sns.boxplot(x=x1, y=y2, order=order, showfliers=True, data=df_train_labeled)
plt.xlabel('')

plt.subplot(324)
sns.boxplot(x=x1, y=y3, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')

plt.subplot(325)
sns.boxplot(x=x1, y=y4, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')
plt.show()

"""#### 6) Cluster 간 분포"""

km = KMeans(n_clusters=4, init="random", random_state=10)
clusts_test1 = km.fit_predict(X_test)

df_test_labeled = df_test.assign(cluster=clusts_test1)
df_test_labeled["cluster"].replace({0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4"}, inplace=True)
df_test_labeled.head()

print(len(df_train_labeled["cluster"]))
(df_train_labeled["cluster"].value_counts() / len(df_train_labeled["cluster"])) * 100

print(len(df_test_labeled["cluster"]))
(df_test_labeled["cluster"].value_counts() / len(df_test_labeled["cluster"])) * 100

"""### Clustering with DB-Scan

#### 1) 최적 파라미터 탐색
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

eps_to_test = [0.01, 0.02]
min_samples_to_test = [10, 20, 75]
print("EPS VALUES:", eps_to_test)
print("MIN_SAMPLES:", min_samples_to_test)


def get_metrics(eps, min_samples, dataset, iter_):
    dbscan_model_ = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan_model_.fit(dataset)

    noise_indices = dbscan_model_.labels_ == -1

    if True in noise_indices:
        neighboors = NearestNeighbors(n_neighbors=6).fit(dataset)
        distances, indices = neighboors.kneighbors(dataset)
        noise_distances = distances[noise_indices, 1:]
        noise_mean_distance = round(noise_distances.mean(), 3)
    else:
        noise_mean_distance = None

    number_of_clusters = len(set(dbscan_model_.labels_[dbscan_model_.labels_ >= 0]))

    print("%3d | Tested with eps = %3s and min_samples = %3s | %5s %4s" % (
        iter_, eps, min_samples, str(noise_mean_distance), number_of_clusters))

    return (noise_mean_distance, number_of_clusters)


results_noise = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),
    columns=min_samples_to_test,
    index=eps_to_test
)

results_clusters = pd.DataFrame(
    data=np.zeros((len(eps_to_test), len(min_samples_to_test))),
    columns=min_samples_to_test,
    index=eps_to_test
)

iter_ = 0

print("ITER| INFO%s |  DIST    CLUS" % (" " * 39))
print("-" * 65)

for eps in eps_to_test:
    for min_samples in min_samples_to_test:
        iter_ += 1

        noise_metric, cluster_metric = get_metrics(eps, min_samples, X_train, iter_)

        results_noise.loc[eps, min_samples] = noise_metric
        results_clusters.loc[eps, min_samples] = cluster_metric

"""#### 2) 최적 모델 기반 클러스터링"""

eps = 0.01
min_samples_to_test = 11

dbscan_model = DBSCAN(eps=eps, min_samples=min_samples_to_test)
DBclusts_train = dbscan_model.fit(X_train)

noise_metric, cluster_metric = get_metrics(eps, min_samples_to_test, X_train, 1)

DBclust = np.concatenate((DBclusts_train.labels_, clusts_test), axis=None)
print(clusts_test.shape, DBclust.shape)
DBclust_series = pd.Series(DBclust)

dbtrainvector = DBclusts_train.labels_

"""#### 3) 시각화

##### PCA
"""

dfDB_PCA = df_PCA.assign(cluster=DBclust_series)
print(dfDB_PCA.columns, dfDB_PCA.shape)

dfDB_PCA['cluster'].value_counts()

dfDB_PCA = dfDB_PCA.dropna()
print(dfDB_PCA.columns, dfDB_PCA.shape)
dfDB_PCA['cluster'].unique()

dfDB_PCA["cluster"].replace({-1: "Cluster 0"}, inplace=True)

dfDB_PCA["cluster"].replace(
    {0: "Cluster 1", 1: "Cluster 2", 2: "Cluster 3", 3: "Cluster 4", 4: "Cluster 5", 42: "Test values"}, inplace=True)
dfDB_PCA.head()

dfDB_PCA['cluster'].value_counts()

db_orderhue = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3', 'Test values']
plt.rc('font', family='AppleGothic')
sns.lmplot(x='PCA_1', y='PCA_2', data=dfDB_PCA, fit_reg=False, legend=False, hue="cluster",
           hue_order=db_orderhue, scatter_kws={"s": 200, "alpha": 0.3})
plt.title('Data clustered with DB-Scan and plotted in 2D after PCA dimensional reduction', fontsize=20)
plt.xlabel('PCA component 1', fontsize=14)
plt.ylabel('PCA component 2', fontsize=14)
plt.legend(fontsize=14)
plt.show()

df_train_labeled = df_train_labeled.assign(dbcluster=dbtrainvector)
df_train_labeled["dbcluster"].replace({-1: "Cluster 1", 0: "Cluster 2", 1: "Cluster 3", 2: "Cluster 4"}, inplace=True)
print(df_train_labeled["dbcluster"].unique())
df_train_labeled.head()

"""##### BoxPlot"""

x2 = df_train_labeled["dbcluster"]
y0 = df_train_labeled[df_train_labeled.columns[0]]
y1 = df_train_labeled[df_train_labeled.columns[1]]
y2 = df_train_labeled[df_train_labeled.columns[2]]
y3 = df_train_labeled[df_train_labeled.columns[3]]
y4 = df_train_labeled[df_train_labeled.columns[4]]
order1 = ["Cluster 1", "Cluster 2", "Cluster 3", "Cluster 4"]

plt.figure(figsize=(16, 16))

plt.xlabel('')
plt.subplot(321)
plt.xticks([])

sns.boxplot(x=x2, y=y0, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')
plt.subplot(322)
plt.xticks([])

sns.boxplot(x=x2, y=y1, order=order, showfliers=True, data=df_train_labeled)
plt.xlabel('')

plt.subplot(323)
plt.xticks([])

sns.boxplot(x=x2, y=y2, order=order, showfliers=True, data=df_train_labeled)
plt.xlabel('')

plt.subplot(324)
sns.boxplot(x=x2, y=y3, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')

plt.subplot(325)
sns.boxplot(x=x2, y=y4, order=order, showfliers=True, data=df_train_labeled)

plt.xlabel('')
plt.show()

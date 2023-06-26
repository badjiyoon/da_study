import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Import and suppress warnings
import warnings

# Settings Warning and Plot Hangul
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'
# 00. 필요한 파이썬 라이브러리 불러오기
# Plotly
import plotly.offline as py

py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBClassifier
from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, classification_report
from sklearn.feature_selection import RFE, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize
from sklearn.decomposition import PCA

import scipy.cluster.hierarchy as shc

from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations

# Import and suppress warnings
import warnings

# Settings Warning and Plot Hangul
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'

print("pandas version: ", pd.__version__)
pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 12)

# 0. 데이터 불러오기
train = pd.read_csv('../../../comFiles/Train.csv')
train.head()
train.shape

# 1. 데이터 탐색
# 1) 데이터 타입
# 컬럼별 데이터 타입 알아보기
train.info()
# 2) 데이터 통계값
# 컬럼별 간단한 통계값 보기
train.describe()
# 3) 변수 간 관계 그래프
# 여러 개의 KDE Plot 생성 준비
f, axes = plt.subplots(2, 2, figsize=(10, 8),
                       sharex=False, sharey=False)

# Plot 색상 설정
s = np.linspace(0, 3, 10)

# 1-1
cmap = sns.cubehelix_palette(start=0.0, light=1, as_cmap=True)
x = train['Customer_rating']
y = train['Prior_purchases']
sns.kdeplot(x=x, y=y, cmap=cmap, shade=True, ax=axes[0, 0])
axes[0, 0].set(title='고객 만족 점수 - 이전 구매 횟수')

# 1-2
cmap = sns.cubehelix_palette(start=0.333333333333, light=1, as_cmap=True)
x = train['Cost_of_the_Product']
y = train['Customer_care_calls']
sns.kdeplot(x=x, y=y, cmap=cmap, shade=True, ax=axes[0, 1])
axes[0, 1].set(title='상품 비용 - 고객 응답 횟수')

# 2-1
cmap = sns.cubehelix_palette(start=1.0, light=1, as_cmap=True)
x = train['Cost_of_the_Product']
y = train['Prior_purchases']
sns.kdeplot(x=x, y=y, cmap=cmap, shade=True, ax=axes[1, 0])
axes[1, 0].set(title='상품 비용 - 이전 구매 횟수')

# 2-2
cmap = sns.cubehelix_palette(start=1.333333333333, light=1, as_cmap=True)
x = train['Customer_care_calls']
y = train['Prior_purchases']
sns.kdeplot(x=x, y=y, cmap=cmap, shade=True, ax=axes[1, 1])
axes[1, 1].set(title='고객 응답 횟수 - 이전 구매 횟수')

f.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
train[train['Mode_of_Shipment'] == 'Flight']['Cost_of_the_Product'].hist(alpha=0.5, color='blue', bins=30,
                                                                         label='Flight')
train[train['Mode_of_Shipment'] == 'Road']['Cost_of_the_Product'].hist(alpha=0.5, color='red', bins=30, label='Road')
plt.xlabel('COST')
plt.legend()
plt.show()
# 4) 결측 값
train.isnull().any()
# 5) 중복 값
# 중복된 항목 수 알아보기
print("중복된 항목 수 :", len(train[train.duplicated()]))

# 2. 데이터 전처리
# 1) 데이터 컬럼명 수정
train = train.rename(columns={'Reached.on.Time_Y.N': 'Reached_on_Time_Y_N'})

# 2. Target Feature 정의
# 1) 데이터 탐색
train.groupby('Reached_on_Time_Y_N').mean()
train.groupby('Reached_on_Time_Y_N').median()

y = train['Reached_on_Time_Y_N'].copy()
y.value_counts()

# 2) 숫자형 변수들과의 관계
hue = 'Reached_on_Time_Y_N'
sns.pairplot(train.select_dtypes(include=np.number), hue=hue)
plt.show()

# 3) 각 변수별 Target Feature 그래프
# 이전 구매횟수 별 Target Feature
sns.countplot(x='Prior_purchases', data=train, palette='plasma', hue='Reached_on_Time_Y_N')
plt.title('Prior_puchase and Reached on-time')
plt.show()

# 고객 평가 점수별 Target Feature
sns.countplot(x="Customer_rating", data=train, palette="plasma", hue="Reached_on_Time_Y_N")
plt.title('Customer Rating and Reached on-time')
plt.show()

# 상품 중요도별 Target Feature
sns.countplot(x="Product_importance", data=train, palette="plasma", hue="Reached_on_Time_Y_N");
plt.title('Product importance and Reached on-time')
plt.show()

# 성별에 따른 Target Feature
sns.countplot(x="Gender", data=train, palette="plasma", hue="Reached_on_Time_Y_N");
plt.title('gender and Reached on-time')
plt.show()

# 창고 구역별 Target Feature
sns.countplot(x="Warehouse_block", data=train, palette="plasma", hue="Reached_on_Time_Y_N",
              order=train['Warehouse_block'].value_counts().index);
plt.title('warehouse block and Reached on-time')
plt.show()

# 운송 수단별 Target Feature
sns.countplot(x="Mode_of_Shipment", data=train, palette="plasma", hue="Reached_on_Time_Y_N");
plt.title('Mode of shipment and Reached on-time')
plt.show()

# 고객 응답 횟수별 Target Feature
sns.countplot(x="Customer_care_calls", data=train, palette="plasma", hue="Reached_on_Time_Y_N");
plt.title('Customer care calls and Reached on-time')
plt.show()

# 4. 데이터 타입별 Feature 변환
# 1) Feature 탐색
# 총 Feature 개수 확인
print(train.info())
# Feature 데이터 타입별 개수 확인
# 데이터 타입별 컬럼 수 확인
dtype_data = train.dtypes.reset_index()
dtype_data.columns = ['Count', 'Column Type']
dtype_data = dtype_data.groupby('Column Type').aggregate('count').reset_index()
print(dtype_data)

# 2) 범주형 Feature#
# * 데이터 확인
# * Feature 별 개수 시각화
# * 날짜/숫자/기간 등으로 변환해야 할 항목이 있는지 확인
# * Feature 별 개수 시각화
# * Feature 의 개수가 인코딩에 적합한가?

# 데이터 확인
train.select_dtypes(include=['object', 'category']).head()

# Feature 제거
# Feature 별 유일한 값 개수 확인
cat_feat = train.select_dtypes('object', 'category').columns.values
train_cat = train[cat_feat].copy()
print(train_cat.nunique().sort_values())
# 유일한 값이 1개인 경우 또는 모든 행의 값이 다른경우 제거한다
# Target Feature가 포함되어 있으면 함께 제거
# Feature 별 개수 시각화
for col in train_cat.columns:
    fig = sns.catplot(x=col, kind='count', data=train_cat, hue=None)
    fig.set_xticklabels(rotation=90)
    plt.show()

# Feature의 개수가 인코딩에 적합한가?
# 인코딩을 했을 경우 메모리 문제가 발생하지 않는가?
# Feature 인코딩
# * LabelEncoder : LabelEncoder 는 선형성을 가지는 머신러닝 기법에 쓰면 좋지 않다
# * OneHotEncoder vs. get_dummies

train_cat_dummies = pd.get_dummies(train_cat)
train_cat_dummies.head(3)

# 3. 숫자형 Feature
# * 데이터 확인
# * Feature 제거
# * Feature Skewness 확인
# 데이터 확인
train.select_dtypes(include=['number']).head()
train.select_dtypes(include=np.number).head()
num_feat = train.select_dtypes('number').columns.values
train_num = train[num_feat].copy()

# Feature 제거
# Feature 별 유일한 값 개수 확인
print(train_num.nunique().sort_values())
print(train.shape)

# 유일한 값이 1개인 경우 또는 모든 행의 값이 다른 경우는 제거한다
# * Target Feature 가 포함되어 있으면 함께 제거
train_num = train_num.drop(['ID', 'Reached_on_Time_Y_N'], axis=1, errors='ignore')
## Feature Skewness 확인
col_attrition_num = train_num.columns.values
for i in range(0, len(col_attrition_num)):
    sns.displot(train_num[col_attrition_num[i]], kde=True)  # kde : kernel density
    plt.show()

# 5. 상관성에 따른 Feature 정제
# 1) 숫자형 Feature
# * 데이터 확인
# * 숫자형 컬럼들 간 Pearson R 상관 계수를 구한다
# * 상관 계수가 0.9 이상인 컬럼들 중 가장 큰 컬럼을 제거해 본다
# * 컬럼들 간 조합 생성 : comb_num_feat = np.array(list(combinations(num_feat, 2)))
# * Pearson R 상관 계수 구하기 : pearsonr(x1, x2)[0]
# 데이터 확인
train_num.nunique().sort_values()

# 상관계수 구하기
# 방법 1
num_feat = train_num.columns.values
comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(train_num[comb[0]], train_num[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)

high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.9]
high_corr_num

# 방법 2
# plotly 안 보이는 문제 해결
import plotly.io as pio

pio.renderers.default = 'notebook_connected'
pio.renderers
pio.renderers.default = 'colab'
pio.renderers

data = [
    go.Heatmap(
        z=train_num.astype(float).corr().values,  # 피어슨 상관계수
        x=train_num.columns.values,
        y=train_num.columns.values,
        colorscale='Viridis',
        reversescale=False,
        opacity=1.0

    )
]

layout = go.Layout(
    title='숫자형 Feature 들의 피어슨 상관계수',
    xaxis=dict(ticks='', nticks=36),
    yaxis=dict(ticks=''),
    width=900, height=700,

)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='labelled-heatmap')

sns.heatmap(train_num.corr(), annot=True, cmap='Pastel1')
plt.show()

# 2) 범주형 Feature
# 데이터 확인
train_cat_dummies.nunique().sort_values()
train_cat_dummies.head()


def cramers_V(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))  # Cross table building
    stat = chi2_contingency(crosstab)[0]  # Keeping of the test statistic of the Chi2 test
    obs = np.sum(crosstab)  # Number of observations
    mini = min(crosstab.shape) - 1  # Take the minimum value between the columns and the rows of the cross table
    return (stat / (obs * mini))


rows = []

for var1 in train_cat_dummies:
    col = []
    for var2 in train_cat_dummies:
        cramers = cramers_V(train_cat_dummies[var1], train_cat_dummies[var2])  # Cramer's V test
        col.append(round(cramers, 2))  # Keeping of the rounded value of the Cramer's V
    rows.append(col)

cramers_results = np.array(rows)
df = pd.DataFrame(cramers_results, columns=train_cat_dummies.columns, index=train_cat_dummies.columns)

df
np.sum(df > 0.9)
# Concat the two dataframes together columnwise
train_final = pd.concat([train["Reached_on_Time_Y_N"], train_num, train_cat_dummies], axis=1)
train_final.head()
target = train["Reached_on_Time_Y_N"]

X = pd.concat([train_num, train_cat_dummies], axis=1)
Y = train["Reached_on_Time_Y_N"]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30)

# 03. 예측분석
# 고객 정보를 활용한 정시 배송 예측
# 1) Logistic Regression
# 모델 생성
model = LogisticRegression()
model.fit(X_train, y_train)
lr_predictions = model.predict(X_test)
# 모델 평가
print("Accuracy score: {}".format(accuracy_score(y_test, lr_predictions)))
print("=" * 60)
print(classification_report(y_test, lr_predictions))

# RFE(Recursive Feature Elimination) 적용
# Backward
rfe = RFE(estimator=model, n_features_to_select=6)
X_rfe = rfe.fit_transform(X, Y)
model.fit(X_rfe, Y)

print(rfe.support_)
print(rfe.ranking_)
print(X.columns[rfe.support_])

# SelectKBest 적용
# Univariate Selection
X_new = SelectKBest(chi2, k=6).fit_transform(X, Y)
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, Y, test_size=0.30)

model = LogisticRegression()
model.fit(X_new_train, y_new_train)
lr_predictions = model.predict(X_new_test)

print("Accuracy score: {}".format(accuracy_score(y_new_test, lr_predictions)))
print("=" * 60)
print(classification_report(y_new_test, lr_predictions))

# 2) SVM
# Feature Scailing
sc_x = StandardScaler()
X_train_sc = sc_x.fit_transform(X_train)
X_test_sc = sc_x.transform(X_test)
# 모델 생성
clf = svm.SVC(kernel='linear')
clf.fit(X_train_sc, y_train)
clf_predictions = clf.predict(X_test_sc)
# 모델 평가
print("Accuracy score: {}".format(accuracy_score(y_test, clf_predictions.round(), normalize=True)))
print("=" * 60)
print(classification_report(y_test, clf_predictions))

# SelectKBest 적용 후 평가
X_norm = MinMaxScaler().fit_transform(X)
X_new = SelectKBest(chi2, k=6).fit_transform(X_norm, Y)

X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, Y, test_size=0.30)

clf.fit(X_new_train, y_new_train)
clf_predictions = clf.predict(X_new_test)

print("Accuracy score: {}".format(accuracy_score(y_new_test, clf_predictions.round(), normalize=True)))
print("=" * 60)
print(classification_report(y_new_test, clf_predictions))

# 3) Random Forest
# 모델 생성
rf = RandomForestRegressor(n_estimators=20, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("Accuracy score: {}".format(accuracy_score(y_test, y_pred.round())))

feat = X.columns.values
imp = rf.feature_importances_
df = pd.DataFrame({'Feature': feat, 'Importance': imp})
df = df.sort_values('Importance', ascending=False)[:10]
sns.barplot(x='Importance', y='Feature', data=df)
plt.show()

trace = go.Scatter(
    y=rf.feature_importances_,
    x=X.columns.values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
        sizeref=1,
        size=13,
        color=rf.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text=X.columns.values
)
data = [trace]

layout = go.Layout(
    autosize=True,
    title='Random Forest Feature Importance',
    hovermode='closest',
    xaxis=dict(
        ticklen=5,
        showgrid=False,
        zeroline=False,
        showline=False
    ),
    yaxis=dict(
        title='Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen=5,
        gridwidth=2
    ),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='scatter2010')

# RFE(Recursive Feature Elimination) 적용
rfe = RFE(estimator=rf, n_features_to_select=6)
# Transforming data using RFE
X_rfe = rfe.fit_transform(X, Y)
# Fitting the data to model
rf.fit(X_rfe, Y)
print(rfe.support_)
print(rfe.ranking_)

# SelectKBest 적용 후 평가
X_new = SelectKBest(chi2, k=6).fit_transform(X, Y)
X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, Y, test_size=0.30)

rf = RandomForestRegressor(n_estimators=20, random_state=0)
rf.fit(X_new_train, y_new_train)
y_pred = rf.predict(X_new_test)

print("Accuracy score: {}".format(accuracy_score(y_new_test, y_pred.round())))
# 4) XGBoost
xgmodel = XGBClassifier()
xgmodel.fit(X_train, y_train)
y_pred = xgmodel.predict(X_test)
print("Accuracy score: {}".format(accuracy_score(y_test, y_pred.round())))

# RFE(Recursive Feature Elimination) 적용
rfe = RFE(estimator=xgmodel, n_features_to_select=6)
X_rfe = rfe.fit_transform(X, Y)
xgmodel.fit(X_rfe, Y)
print(rfe.support_)
print(rfe.ranking_)

# SelectKBest 적용 후 평가
X_new = SelectKBest(chi2, k=6).fit_transform(X, Y)

X_new_train, X_new_test, y_new_train, y_new_test = train_test_split(X_new, Y, test_size=0.30)

xgmodel = XGBClassifier()
xgmodel.fit(X_new_train, y_new_train)
y_pred = xgmodel.predict(X_new_test)

print("Accuracy score: {}".format(accuracy_score(y_new_test, y_pred.round())))

# 04. 군집분석
# 군집 분석을 활용한 고객 Segmentation
# 1) K-Means Clustering
# Elbow 방법 적용
clustdata = train_final[(train_final.Reached_on_Time_Y_N == 1)]

wcss = []
for i in range(1, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(clustdata)
    wcss.append(kmeans.inertia_)

# WCSS는 클러스터의 각 구성원과 중심 사이의 거리 제곱의 합
plt.plot(range(1, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('클러스터 수')
plt.ylabel('WCSS')
plt.show()

km = KMeans(n_clusters=2, init='k-means++', n_init=10)
c = km.fit_predict(clustdata)
c
clustdata["Cluster"] = c
clustdata.head()

clustdata.groupby(['Cluster']).mean()
# 군집화된 데이터 Scailing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(clustdata)

X_normalized = normalize(X_scaled)
X_normalized = pd.DataFrame(X_normalized)

# PCA 적용 (n=2)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(X_normalized)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['P1', 'P2']

c1 = km.fit_predict(X_principal)
c1

result = pd.DataFrame({'P1': X_principal.iloc[:, 0], 'P2': X_principal.iloc[:, 1]})
result['Cluster'] = pd.Series(c1, index=result.index)
result.head()

# 주성분-클러스터 그래프
facet = sns.lmplot(data=result, x='P1', y='P2', hue='Cluster', fit_reg=False, legend=True, legend_out=True)
result['Cluster'].value_counts()

# PCA 후 Elbow 방법 적용
wcss = []
for i in range(2, 15):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=0)
    kmeans.fit(X_principal)
    wcss.append(kmeans.inertia_)

plt.plot(range(2, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

km = KMeans(n_clusters=4, init='k-means++', n_init=10)
c = km.fit_predict(X_principal)
c

result = pd.DataFrame({'P1': X_principal.iloc[:, 0], 'P2': X_principal.iloc[:, 1]})
result['Cluster'] = pd.Series(c, index=result.index)
result.head()

# 새로운 주성분-클러스터 그래프
# Visualizing the clustering
plt.figure(figsize=(6, 6))
plt.scatter(result.iloc[:, 0], result.iloc[:, 1], c=result['Cluster'], cmap='rainbow')
plt.show()

facet = sns.lmplot(data=result, x='P1', y='P2', hue='Cluster', fit_reg=False, legend=True, legend_out=True)
plt.show()

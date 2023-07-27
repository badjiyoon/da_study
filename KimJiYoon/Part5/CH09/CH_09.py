# Part4. [실습7] 자동차 제조 테스트 공정 시간 예측하기
############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import combinations
from sklearn import preprocessing
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.cluster import FeatureAgglomeration

from sklearn.random_projection import GaussianRandomProjection
from sklearn.random_projection import SparseRandomProjection
from sklearn.linear_model import ElasticNetCV, LassoLarsCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Ridge

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.utils import check_array
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

import warnings

# Settings Warning and Plot Hangul
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')

color = sns.color_palette()
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'

seed = 40

# 01. 데이터 소개 및 분석프로세스 수립 : "강의자료 → Ch09. [실습7] 자동차 제조 테스트 공정 시간 예측하기" 참고
# 02. 데이터 준비를 위한 EDA 및 전처리
# 0. 데이터 불러오기
train_df = pd.read_csv("../../../comFiles/ch_09_train.csv")
test_df = pd.read_csv("../../../comFiles/ch_09_test.csv")
print("Train shape : ", train_df.shape)
print("Test shape : ", test_df.shape)
train_df.head()
# > Feature 데이터
# 1. **ID**
# 2. **y** : Target Feature
# 3. **X0-X385**
# Target Feature
# scatter plot
plt.figure(figsize=(8, 6))
# np.sort -> 시간의 순서에 따라 정렬
plt.scatter(range(train_df.shape[0]), np.sort(train_df.y.values))
plt.xlabel('index', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title("Target Variable: 'y'", fontsize=15)
plt.show()
# Histogram
# 가장 큰값
ulimit = 180
train_df['y'].loc[train_df['y'] > ulimit] = ulimit
plt.figure(figsize=(12, 8))
sns.distplot(train_df.y.values, bins=50, kde=False)
plt.xlabel('y value', fontsize=12)
plt.title("Histogram of Target Feature", fontsize=15)
plt.show()

print('최소값: {} 최대값: {} 평균값: {} 표준편차: {}'.format(min(train_df['y'].values), max(train_df['y'].values),
                                                train_df['y'].values.mean(), train_df['y'].values.std()))
print('180보다 큰 숫자들 개수: {}'.format(np.sum(train_df['y'].values > 180)))

# 데이터 탐색
dtype_df = train_df.dtypes.reset_index()
dtype_df.columns = ["Count", 'Column Type']
dtype_df.groupby("Column Type").aggregate("count").reset_index()
dtype_df.loc[:10, :]

# 결측값
missing_df = train_df.isnull().sum(axis=0).reset_index()
missing_df.columns = ["column_name", "missing_count"]
missing_df = missing_df.loc[missing_df["missing_count"] > 0]
missing_df = missing_df.sort_values(by="missing_count")
missing_df

# 결측값 확인
cols = [c for c in train_df.columns if "X" in c]
print('Number of features: {}'.format(len(cols)))
print('Feature types:')
train_df[cols].dtypes.value_counts()

counts = [[], [], []]
for c in cols:
    typ = train_df[c].dtype
    # 각각의 유니크한 길이를 확인한다
    uniq = len(np.unique(train_df[c]))
    if uniq == 1:
        counts[0].append(c)
    elif uniq == 2 and typ == np.int64:
        counts[1].append(c)
    else:
        counts[2].append(c)

print('Feature 값이 1개인 경우 : {} Feature 값이 2개인 경우: {} 범주형 Feature 인 경우: {}\n'.format(*[len(c) for c in counts]))
print('Feature 값이 1개인 경우: ', counts[0])
print('Feature 값이 2개인 경우: ', counts[2])

unique_values_dict = {}
for col in train_df.columns:
    if col not in ["ID", "y", "X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
        unique_value = str(np.sort(train_df[col].unique()).tolist())
        tlist = unique_values_dict.get(unique_value, [])
        tlist.append(col)
        unique_values_dict[unique_value] = tlist[:]
for unique_val, columns in unique_values_dict.items():
    print("컬럼에 존재하는 유일한 값들 : ", unique_val)
    print(columns)
    print("--------------------------------------------------")

# 범주형 Feature
cat_feat = counts[2]
train_df[cat_feat].head()

binary_means = [np.mean(train_df[c]) for c in counts[1]]
binary_names = np.array(counts[1])[np.argsort(binary_means)]
binary_means = np.sort(binary_means)

fig, ax = plt.subplots(1, 3, figsize=(12, 30))
ax[0].set_ylabel('Feature 이름')
ax[1].set_title('유일한 값 개수가 2개인 변수들의 평균')
for i in range(3):
    names, means = binary_names[i * 119:(i + 1) * 119], binary_means[i * 119:(i + 1) * 119]
    ax[i].barh(range(len(means)), means, color=color[2])
    ax[i].set_xlabel('평균값')
    ax[i].set_yticks(range(len(means)))
    ax[i].set_yticklabels(names, rotation='horizontal')
plt.show()

# 03. 머신러닝 모델링
# Baseline Model 1: xgboost model
# Label Encoding

for f in ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[f].values))  # Fit label encoder
    train_df[f] = lbl.transform(list(train_df[f].values))  # Transform labels to normalized encoding.

# 데이터 준비
train_y = train_df['y'].values
train_X = train_df.drop(["ID", "y"], axis=1)


# 모델 생성
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100, feval=xgb_r2_score, maximize=True)

# Feature Importances
fig, ax = plt.subplots(figsize=(12, 18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

from sklearn import ensemble

model = ensemble.RandomForestRegressor(n_estimators=200, max_depth=10, min_samples_leaf=4, max_features=0.2, n_jobs=-1,
                                       random_state=0)
model.fit(train_X, train_y)

# Feature Importances
feat_names = train_X.columns.values
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = np.argsort(importances)[::-1][:20]

plt.figure(figsize=(12, 12))
plt.title("Feature importances")
plt.bar(range(len(indices)), importances[indices], color="r", align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical')
plt.xlim([-1, len(indices)])
plt.show()

# 4. 다양한 차원 축소 기법을 활용한 Feature Selection
# 일반적인 적용
# 차원 축소 기법의 종류

# * Principal Component Analysis [PCA]
# * Independent Component Analysis [ICA]
# * Truncated SVD [TSVD]
# * Gaussian Random Projection [GRP]
# * Sparse Random Projection [SRP]
# * Non-negative Matrix factorization [NMF]
# * Feature Agglomeration [FAG]

# 데이터 준비
train = pd.read_csv("../../../comFiles/ch_09_train.csv")
test = pd.read_csv("../../../comFiles/ch_09_test.csv")

y_train = train['y']
train = train.drop('y', axis=1)

# Label Encoding
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))
        train[c] = lbl.transform(list(train[c].values))
        test[c] = lbl.transform(list(test[c].values))

# 차원 축소 방법 적용
n_comp = 12

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train)
tsvd_results_test = tsvd.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)
srp_results_train = srp.fit_transform(train)
srp_results_test = srp.transform(test)

# NMF
nmf = NMF(n_components=n_comp, init='nndsvdar', random_state=420)
nmf_results_train = nmf.fit_transform(train)
nmf_results_test = nmf.transform(test)

# FAG
fag = FeatureAgglomeration(n_clusters=n_comp, linkage='ward')
fag_results_train = fag.fit_transform(train)
fag_results_test = fag.transform(test)

dim_reds = list()
train_pca = pd.DataFrame()
test_pca = pd.DataFrame()

train_ica = pd.DataFrame()
test_ica = pd.DataFrame()

train_tsvd = pd.DataFrame()
test_tsvd = pd.DataFrame()

train_grp = pd.DataFrame()
test_grp = pd.DataFrame()

train_srp = pd.DataFrame()
test_srp = pd.DataFrame()

train_nmf = pd.DataFrame()
test_nmf = pd.DataFrame()

train_fag = pd.DataFrame()
test_fag = pd.DataFrame()

for i in range(1, n_comp + 1):
    train_pca['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test_pca['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train_ica['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test_ica['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train_tsvd['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test_tsvd['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train_grp['grp_' + str(i)] = grp_results_train[:, i - 1]
    test_grp['grp_' + str(i)] = grp_results_test[:, i - 1]

    train_srp['srp_' + str(i)] = srp_results_train[:, i - 1]
    test_srp['srp_' + str(i)] = srp_results_test[:, i - 1]

    train_nmf['nmf_' + str(i)] = nmf_results_train[:, i - 1]
    test_nmf['nmf_' + str(i)] = nmf_results_test[:, i - 1]

    train_fag['fag_' + str(i)] = fag_results_train[:, i - 1]
    test_fag['fag_' + str(i)] = fag_results_test[:, i - 1]

dim_reds.append(('pca', train_pca, test_pca))
dim_reds.append(('ica', train_ica, test_ica))
dim_reds.append(('tsvd', train_tsvd, test_tsvd))
dim_reds.append(('grp', train_grp, test_grp))
dim_reds.append(('srp', train_srp, test_srp))
dim_reds.append(('nmf', train_nmf, test_nmf))
dim_reds.append(('fag', train_fag, test_fag))

#### 복수의 차원 축소 기법 적용에 따른 성능 실험
combs = [combinations(dim_reds, i + 1) for i in range(0, len(dim_reds))]

dr_scores = list()
for c1 in combs:
    for c2 in c1:
        train_, test_, id_ = list(), list(), list()
        for k in c2:
            train_.append(k[1])
            test_.append(k[2])
            id_.append(k[0])

        train_x = train.reset_index(drop=True)
        train_.append(train_x)
        test_.append(test)

        train_ = pd.concat(train_, axis=1)
        test_ = pd.concat(test_, axis=1)

        # ============================ DecisionTree Model =======================
        #         model = DecisionTreeRegressor(max_depth=3, min_samples_split=11, presort=False, random_state=1729)
        #         model.fit(train_, y_train)
        #         c_score = r2_score(y_train, model.predict(train_))
        # normalize=True
        # ============================ ElasticNet model =======================
        model = ElasticNet(alpha=0.014, tol=0.11, l1_ratio=0.99999999,
                           fit_intercept=False, warm_start=True,
                           copy_X=True, precompute=False, positive=False, max_iter=60)
        model.fit(train_, y_train)
        c_score = r2_score(y_train, model.predict(train_))

        # ============================ Ridge model =============================
        #         model = Ridge()
        #         model.fit(train_, y_train)
        #         c_score = r2_score(y_train, model.predict(train_))

        # ================================ lightgbm model =======================
        #         lgb_params = {
        #         'num_iterations': 200,
        #         'learning_rate': 0.045,
        #         'max_depth': 3,
        #         'bagging_fraction': 0.93,
        #         'metric': 'l2_root',
        #         }

        #         dtrain = lgb.Dataset(train_, y_train)
        #         num_round = 1200
        #         model = lgb.train(lgb_params, dtrain, num_round)
        #         c_score = r2_score(y_train, model.predict(train_))

        # ================================= xgboost model ============================
        #         xgb_params = {
        #         'n_trees': 520,
        #         'eta': 0.0045,
        #         'max_depth': 4,
        #         'subsample': 0.93,
        #         'objective': 'reg:linear',
        #         'eval_metric': 'rmse',
        #         'base_score': np.mean(y_train),
        #         }

        #         dtrain = xgb.DMatrix(train_, y_train)

        #         num_boost_rounds = 1250
        #         model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_rounds)
        #         c_score = r2_score(y_train, model.predict(dtrain))

        dr_scores.append((','.join(id_), c_score))

# baseline scoring for comparision
model = ElasticNet(alpha=0.014, tol=0.11, l1_ratio=0.99999999,
                   fit_intercept=False, warm_start=True,
                   copy_X=True, precompute=False, positive=False, max_iter=60)
model.fit(train, y_train)
full_score = r2_score(y_train, model.predict(train))

dr_scores.append(('baseline', full_score))

# Plotting the graph
# > Please open the plots in a separate tab for better labels and clarity.
for dr_score in dr_scores:
    if dr_score[1] < 0:
        dr_score = (dr_score[0], 0)

x_axis = [c[0] for c in dr_scores]
y_axis = [c[1] for c in dr_scores]
fig, ax = plt.subplots(figsize=(22, 10))
plt.plot(y_axis)
ax.set_xticks(range(len(x_axis)))
ax.set_xticklabels(x_axis, rotation='vertical')
plt.show()

#### 모델 성능 비교
sorted_id = np.argsort(y_axis)
print("점수 하위 7개 차원 축소 기법 조합 : {}".format(np.array(x_axis)[sorted_id[:7]]))
print("점수 상위 7개 차원 축소 기법 조합 : {}".format(np.array(x_axis)[sorted_id[-7:]]))

print("\n\n가장 높은 점수를 가지는 조합 : {}".format(np.array(x_axis)[sorted_id[-1]]))

from sklearn.decomposition import PCA, FastICA, TruncatedSVD

# Dimensionality reduction techniques
n_comp = 12

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)
grp_results_train = grp.fit_transform(train)
grp_results_test = grp.transform(test)

# PCA
pca = PCA(n_components=n_comp, random_state=420)
pca2_results_train = pca.fit_transform(train)
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=420)
ica2_results_train = ica.fit_transform(train)
ica2_results_test = ica.transform(test)

for i in range(1, n_comp + 1):
    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

y_mean = np.mean(y_train)

import xgboost as xgb

# Prepare dict of params for xgboost model.
xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 6,
    'subsample': 0.5,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,  # base prediction = mean(target)
    'silent': 1}

# Creating DMatrices for Xgboost training
dtrain = xgb.DMatrix(train, y_train)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=700, verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# Train model
model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
#### Feature Importances
fig, ax = plt.subplots(figsize=(12, 18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()
from sklearn.metrics import r2_score

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('XGB_test_results.csv', index=False)
output = pd.DataFrame({'id': test['ID'].astype(np.int32), 'y': y_pred})
plt.figure(figsize=(12, 8))
sns.distplot(output.y.values, bins=50, kde=False)
plt.xlabel('Predicted Time on Test Data', fontsize=13)
plt.ylabel('Test Data', fontsize=13)
plt.title('Histogram of Predicted Time on Test Data', fontsize=15)
plt.show()


### Stacked Regression model 적용
#### 각 Estimator 의 결과를 합칠 클래스 생성
# * BaseEstimator : https://scikit-learn.org/stable/modules/generated/sklearn.base.BaseEstimator.html
# * TransformerMixin : https://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html
#
# > 파이썬 클래스, 상속, 멤버 변수, 생성자 등의 기초 지식 필요

class StackingEstimator(BaseEstimator, TransformerMixin):
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params):
        self.estimator.fit(X, y, **fit_params)
        return self

    """
    1. np.hstack : 두 배열을 가로로 이어붙이는 함수
    2. predict_proba 의 출력은 각 클래스에 대한 확률
    3. issubclass(subclass, superclass) -> subclass가 superclass의 자식 클래스인지, 다시 말해 subclass가 superclass를 상속받는지 판단해 True, False 를 반환
    4. hasattr(self.estimator, 'predict_proba') -> self.estimator(우리의 머신러닝 모델) 에 'predict_proba' 라는 멤버가 있는지 판단해 True, False 를 반환
    """

    def transform(self, X):
        X = check_array(X)
        X_transformed = np.copy(X)

        if issubclass(self.estimator.__class__, ClassifierMixin) and hasattr(self.estimator, 'predict_proba'):
            # X 에 각 클래스에 대한 확률을 이어 붙여준다.
            X_transformed = np.hstack((self.estimator.predict_proba(X), X))

        # 위의 X_transformed 에 예측 클래스를 이어 붙여준다.
        X_transformed = np.hstack((np.reshape(self.estimator.predict(X), (-1, 1)), X_transformed))

        return X_transformed

train = pd.read_csv("../../../comFiles/ch_09_train.csv")
test = pd.read_csv("../../../comFiles/ch_09_test.csv")

#### Label Encoding
for c in train.columns:
    if train[c].dtype == 'object':
        lbl = LabelEncoder()
        lbl.fit(list(train[c].values) + list(test[c].values))  # Fit label encoder
        train[c] = lbl.transform(list(train[c].values))  # Transform labels to normalized encoding.
        test[c] = lbl.transform(list(test[c].values))  # Transform labels to normalized encoding.

# 데이터 준비
train_y = train['y'].values
y_mean = np.mean(train_y)
id_test = test['ID'].values
train = train.drop(["ID"], axis=1)
test = test.drop(["ID"], axis=1)


def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)


usable_columns = list(set(train.columns) - set(['y']))

n_comp = 15

# tSVD
tsvd = TruncatedSVD(n_components=n_comp, random_state=420)
tsvd_results_train = tsvd.fit_transform(train.drop(["y"], axis=1))
tsvd_results_test = tsvd.transform(test)
# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))
pca2_results_test = pca.transform(test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(train.drop(["y"], axis=1))
ica2_results_test = ica.transform(test)

# GRP
grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=42)
grp_results_train = grp.fit_transform(train.drop(["y"], axis=1))
grp_results_test = grp.transform(test)

# SRP
srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=42)
srp_results_train = srp.fit_transform(train.drop(["y"], axis=1))
srp_results_test = srp.transform(test)

for i in range(1, n_comp + 1):
    train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    train['pca_' + str(i)] = pca2_results_train[:, i - 1]
    test['pca_' + str(i)] = pca2_results_test[:, i - 1]

    train['ica_' + str(i)] = ica2_results_train[:, i - 1]
    test['ica_' + str(i)] = ica2_results_test[:, i - 1]

    train['grp_' + str(i)] = grp_results_train[:, i - 1]
    test['grp_' + str(i)] = grp_results_test[:, i - 1]

    train['srp_' + str(i)] = srp_results_train[:, i - 1]
    test['srp_' + str(i)] = srp_results_test[:, i - 1]

final_train = train[usable_columns].values
final_test = test[usable_columns].values

import xgboost as xgb

xgb_params = {
    'n_trees': 500,
    'eta': 0.005,
    'max_depth': 5,
    'subsample': 0.9,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean,
    'silent': 1}

# DMatrix : Numpy 입력 파라미터를 받아서 만들어지는 XGBoost 전용 데이터 --> Input : Feature 데이터, Label 데이터
dtrain = xgb.DMatrix(train.drop(["y"], axis=1), train_y)
dtest = xgb.DMatrix(test)

# xgboost & cross-validation
cv_result = xgb.cv(xgb_params, dtrain, num_boost_round=1000, verbose_eval=50, show_stdv=False)

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
y_pred = model.predict(dtest)

from sklearn.pipeline import make_pipeline, make_union

Stacked_pipeline = make_pipeline(
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=GradientBoostingRegressor(learning_rate=0.001, loss="huber", max_features=0.55,
                                                          min_samples_leaf=18,
                                                          min_samples_split=14, subsample=0.7)),
    LassoLarsCV())

Stacked_pipeline.fit(final_train, train_y)
predictions = Stacked_pipeline.predict(final_test)

print('R2 score on train data:')
print(r2_score(train_y, Stacked_pipeline.predict(final_train) * 0.2855 + model.predict(dtrain) * 0.7145))

sub = pd.DataFrame()
sub['ID'] = id_test
sub['y'] = y_pred * 0.75 + predictions * 0.25
sub.to_csv('stacked_model_pred.csv', index=False)
#### Feature Importances
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)
plt.show()

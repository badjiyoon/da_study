# 01. 데이터 소개 및 분석프로세스 수립
# : "강의자료 → Ch11. [실습9] 가스 터빈 추진 플랜트 제조 공정의 부식 예측" 참고
# 02. 데이터 준비를 위한 EDA 및 전처리
# 0. 데이터 불러오기
############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.inspection import permutation_importance

import warnings

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = "AppleGothic"

naval_df = pd.read_csv("../../../comFiles/navalplantmaintenance.csv", delim_whitespace=True, header=None)
naval_df.head()
naval_df.shape
# 1. 데이터 탐색
# 1) 컬럼명 생성
naval_df.columns = ["lever_position", "ship_speed", "gt_shaft", "gt_rate", "gg_rate", "sp_torque", "pp_torque",
                    "hpt_temp", "gt_c_i_temp", "gt_c_o_temp", "hpt_pressure", "gt_c_i_pressure", "gt_c_o_pressure",
                    "gt_exhaust_pressure", "turbine_inj_control", "fuel_flow", "gt_c_decay", "gt_t_decay"]
# 2) 결측값
# 왜 썼는지 나눴는지에 대한 설명이 없음
100 * naval_df.isna().sum() / len(naval_df)
naval_df = naval_df.dropna()
naval_df.head()
# 3) 데이터 통계값
naval_df.describe()

# 2. Feature 변환
# 1) Feature 탐색
naval_df.info()

# Feature별 유일한 값 개수 확인
# 각 컬럼을 for로 가져와서 유니크한 값을 본다.
[(f"{col} :", len(naval_df[col].unique())) for col in naval_df]
print(naval_df.nunique().sort_values())
# Feature 제거
# we can drop gt_c_i_pressure and gt_c_i_temp as they have only 1 unique value, and thus not conributing to our dataset
naval_df = naval_df.drop(['gt_c_i_pressure', 'gt_c_i_temp'], axis=1)
# 3. Target Feature 정의
# 1) 데이터 확인
naval_df.gt_c_decay.unique()
# 2) 그래프
plt.figure(figsize=(10, 6))
plt.plot(naval_df.index, naval_df.gt_c_decay, ".-")
plt.xlabel("sampleID")
plt.ylabel("gt_c_decay")
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(naval_df.index, naval_df.gt_t_decay, ".-")
plt.xlabel("sampleID")
plt.ylabel("gt_t_decay")
plt.show()

# 4. Feature 정제
# 1) 분포 및 이상치 확인
plt.figure(figsize=(22, 20))
icount = 1
for col in naval_df.columns:
    plt.subplot(4, 4, icount)
    sns.boxplot(naval_df[col], orient="v")
    icount += 1
plt.show()

plt.figure(figsize=(20, 20))
icount = 1
for col in naval_df.columns:
    plt.subplot(4, 4, icount)
    sns.distplot(naval_df[col])
    icount += 1
plt.show()

sns.pairplot(naval_df)
plt.show()

# 2) 상관성 확인
# 선형성 : 범위를 줄여서 확인
sns.pairplot(naval_df[naval_df.columns[2:-2]])
plt.show()

# 상관계수
plt.figure(figsize=(15, 10))
cols = naval_df.corr().index
corr_mat = np.corrcoef(naval_df[cols].values.T)
sns.set(font_scale=1)
hm = sns.heatmap(corr_mat, annot=True, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

# 5. PCA를 활용한 Feature Selection
# 1) 데이터 준비
np.random.seed(0)
df_train_navel, df_test_navel = train_test_split(naval_df, train_size=0.7, test_size=0.3, random_state=100)

y_train_c = df_train_navel.pop('gt_c_decay')
y_train_t = df_train_navel.pop('gt_t_decay')
X_train = df_train_navel

y_test_c = df_test_navel.pop('gt_c_decay')
y_test_t = df_test_navel.pop('gt_t_decay')
X_test = df_test_navel

X_train.shape

tr_scaled_features = StandardScaler().fit_transform(X_train.values)
X_train = pd.DataFrame(tr_scaled_features, index=X_train.index, columns=X_train.columns)

tt_scaled_features = StandardScaler().fit_transform(X_test.values)
X_test = pd.DataFrame(tt_scaled_features, index=X_test.index, columns=X_test.columns)

# PCA 적용
# > PCA 모델링 순서
# * PCA.fit() : 주성분 탐색
# * PCA.transform() : 새로운 주성분으로 데이터 변환
pca = PCA(random_state=42)
pca.fit(X_train)

plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
plt.show()

var_cumu = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(var_cumu) + 1), var_cumu)
plt.grid()
plt.show()

print("no. of Components  Variance accounted")
for i in range(2, 8):
    s = ("      " + str(i) + "             " + str(100 * var_cumu[i]));
    print(s)

pca.components_


# numpy.argmax : 다차원 배열에서 차원에 따라 가장 큰 값의 인덱스들을 반환하는 함수

def getPCAMostImportantFeat(model, initial_feature_names):
    # 총 component 개수
    n_pcs = model.components_.shape[0]

    # 각 component 의 중요도 값들 중 가장 큰 값의 인덱스를 반환
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]

    # 컬럼명 중 위에서 구한 인덱스들에 해당하는 것들만 반환
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # 인덱스 번호와 해당 컬럼명으로 dictionary 생성
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    df = pd.DataFrame(dic.items())
    return df


pca_grid_df = getPCAMostImportantFeat(pca, X_train.columns)
pca_grid_df

# > PCA (n_components=4)
# PCA
pca_4_cpnt = PCA(n_components=4, random_state=42)
# fitting
navel_pca_data = pca_4_cpnt.fit_transform(X_train)

cmp_lst = ['PC' + str(i) for i in range(1, 5)]
cmp_lst
navel_pca_X = pd.DataFrame(navel_pca_data, columns=cmp_lst)
navel_pca_X

navel_pca_X.reset_index(drop=True, inplace=True)

# Transform
navel_pca_data_test = pca_4_cpnt.transform(X_test)
navel_pca_test_X = pd.DataFrame(navel_pca_data_test, columns=cmp_lst)

## 다양한 Regressor 를 활용한 모델 생성과 Hypertuning
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

models = {'LinearRegression': LinearRegression(),
          'RandomForestRegressor': RandomForestRegressor(),
          'KNeighborsRegressor': KNeighborsRegressor(),
          'DecisionTreeRegressor': DecisionTreeRegressor(),
          'BaggingRegressor': BaggingRegressor(),
          'XGBRegressor': XGBRegressor()}

params = {'LinearRegression': [{'fit_intercept': [True, False], 'copy_X': [True, False]}],
          'RandomForestRegressor': [{'n_estimators': [50, 60, 80]}],
          'KNeighborsRegressor': [{'n_neighbors': [2, 3, 4, 5, 6]}],
          'DecisionTreeRegressor': [{'max_depth': [2, 4, 6, 8, 10, 12]}],
          'BaggingRegressor': [{'base_estimator': [None, GradientBoostingRegressor(), KNeighborsRegressor()],
                                'n_estimators': [20, 50, 100]}],
          'XGBRegressor': [{'n_estimators': [50, 500]}]
          }

pca_grid_df.head()
x_pca_cols = pca_grid_df.iloc[:, 1].tolist()
x_pca_cols
# 2. 모델 생성
# > 하나의 함수로 여러 Regressor 들에 대해 GridSearchCV 적용

important_features_list = []
plt.figure(figsize=(20, 12))


def runregressors(X_train, Y_train, X_test, Y_test):
    i_count = 0
    # 총 3 x 2, 6개의 그래프를 생성
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))

    # 각 평가 지표를 저장할 변수들 생성
    result_name = []
    result_summary1 = []
    result_mae = []
    result_mse = []
    result_exp_var = []
    result_r2_score = []
    result_ac_score = []

    for name in models.keys():

        # estimator 와 parameter 를 가져온다
        est = models[name]
        est_params = params[name]

        """
        최적 파라미터 탐색
        """
        # GridSearchCV 생성하여 fitting (cv=5)
        gscv = GridSearchCV(estimator=est, param_grid=est_params, cv=5)
        gscv.fit(X_train, Y_train)

        # 가장 좋은 성능 지표를 저장한다
        msg1 = str(gscv.best_estimator_)
        result_summary1.append(msg1)
        result_name.append(name)

        # 베스트 모델로 predict
        y_pred = gscv.predict(X_test)

        """
        모델 평가
        """
        ascore = gscv.best_estimator_.score(X_test, Y_test)

        # explained_variance_score 적용
        score = explained_variance_score(Y_test, y_pred)

        # mean_absolute_error 적용
        mae = mean_absolute_error(Y_test, y_pred)

        # mean_squared_error 적용
        mse = mean_squared_error(Y_test, y_pred)

        # r2_score 적용
        r2 = r2_score(Y_test, y_pred)

        msg2 = "%s: %f (%f)" % (name, score * 100, mae * 100)
        print(msg2)

        result_mse.append(mse)
        result_mae.append(mae)
        result_exp_var.append(score)
        result_r2_score.append(r2)
        result_ac_score.append(ascore)

        """
        Feature Importance

        RandomForestRegressor, DecisionTreeRegressor, XGBRegressor : 자체 feature_importances_ 내장
        LinearRegression : coef_
        KNeighborsRegressor : permutation_importance ---> importances_mean
        BaggingRegressor : gscv.best_estimator_ 의 feature_importances 

        """

        if name == "LinearRegression":
            # coefficient 가져오기
            important_features = pd.Series(gscv.best_estimator_.coef_, index=x_pca_cols[:4])

        elif name == "KNeighborsRegressor":
            # permutation_importance 적용
            results = permutation_importance(gscv.best_estimator_, X_train, Y_train, scoring='neg_mean_squared_error')
            # importance
            important_features = pd.Series(results.importances_mean, index=x_pca_cols[:4])

        elif name == "BaggingRegressor":
            feature_importances = np.mean([tree.feature_importances_ for tree in gscv.best_estimator_], axis=0)
            important_features = pd.Series(feature_importances, index=x_pca_cols[:4])

        else:
            important_features = pd.Series(gscv.best_estimator_.feature_importances_, index=x_pca_cols[:4])
        important_features_list.append(important_features)

        col = i_count % 2
        row = i_count // 2
        ax[row][col].scatter(Y_test, y_pred)
        ax[row][col].plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=2)
        ax[row][col].set_xlabel('실제값')
        ax[row][col].set_ylabel('예측값')
        ax[row][col].set_title(msg2)
        i_count += 1

    plt.show()

    result_summary_list = pd.DataFrame({'name': result_name,
                                        'best_estimator': result_summary1,
                                        'R2': result_r2_score,
                                        'MAE': result_mae,
                                        'MSE': result_mse,
                                        'explained variance score': result_exp_var,
                                        'accuracy': result_ac_score})
    return result_summary_list


# 컴프레서 부식 예측
result_summary_list = runregressors(navel_pca_X, y_train_c, navel_pca_test_X, y_test_c)

for i in range(0, 4):
    important_features_list[0][i] = abs(important_features_list[0][i])

fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))
i_count = 0
nm = result_summary_list.name.to_list()
for imp_fea in important_features_list:
    col = i_count % 2
    row = i_count // 2
    imp_fea.sort_values().plot(kind='barh', ax=ax[row][col])
    ax[row][col].set_title(nm[i_count])
    i_count += 1

plt.show()
result_summary_list

# 가스터빈 부식 예측
result_summary_list_t = runregressors(navel_pca_X, y_train_t, navel_pca_test_X, y_test_t)
result_summary_list_t

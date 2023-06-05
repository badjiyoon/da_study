# * [accepted_2007_to_2018Q4.csv] : https://www.kaggle.com/wordsforthewise/lending-club
# * [LCDataDictionary.xlsx] : https://data.world/lpetrocelli/lendingclub-loan-data-2017-q-1
# 승인데이터와 사전데이터 다운로드
# Part5. [실습2] 대출 상품 투자 위험도 줄이기
# 01. 데이터 소개 및 분석프로세스 수립
# : '강의자료 → Ch04. [실습2] 대출 상품 투자 위험도 줄이기' 참고
# import matplotlib.font_manager as fm
# path = '/usr/share/fonts/truetype/nanum/NanumGothicEco.ttf'
# font_name = fm.FontProperties(fname=path, size=10).get_name()
# plt.rc('font', family=font_name)
# fm._rebuild()

from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'

############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import pandas as pd
import numpy as np

print(np.__version__)

import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, precision_recall_curve
from catboost import Pool, CatBoostClassifier

from scipy.stats import pearsonr, chi2_contingency
from itertools import combinations
from statsmodels.stats.proportion import proportion_confint

# 데이터 컬럼의 자세한 사항
# https://freelife1191.github.io/dev/2018/05/01/dev-data_analysis-16.python_data_analysis/

#### 1) 대출 승인내역 불러오기
# * issue_d 컬럼은 parse_dates 사용
# * infer_datetime_format = True
# * reset_index(drop=True) 사용

### 0. 데이터 불러오기
# issue_d : 대출금 조달 날짜
data = pd.read_csv('../../../comFiles/accepted_2007_to_2018Q4.csv', parse_dates=['issue_d'], infer_datetime_format=True)
# 2260701 x 151
print('data', data)
# 날짜별 대출 처리 1년치 데이터만 가져옴
data = data[(data.issue_d >= '2018-01-01 00:00:00') & (data.issue_d < '2019-01-01 00:00:00')]
data = data.reset_index(drop=True)
data.head()

# 고위험/고금리 대출 중 양호한 대출 분류 모델을 만들어 예측

#### 2) 대출용어 사전 불러오기
# * pandas 의 read_excel 사용 / Excel Sheet 중 2번째(sheet_name=1)을 불러온다.

browse_notes = pd.read_excel('../../../comFiles/LCDataDictionary.xlsx', sheet_name=1)
browse_notes.head()

# 1. 데이터 전처리
# 기준정보 데이터와 대출승인 데이터의 정합성 맞추기
# 1) 대출용어 사전 결측값 제거
browse_notes['BrowseNotesFile'].dropna().values
browse_feat = browse_notes['BrowseNotesFile'].dropna().values
browse_feat

#### 2) 대출용어 사전과 대출승인 데이터 문자열 규칙 맞추기
# * 대문자, 숫자 앞에 '_' 붙여주고 모두 소문자 변환
#   > re.sub('(?<![0-9_])(?=[A-Z0-9])', '_', x).lower()
# * 공백 처리
#   > .strip()
browse_feat = [re.sub('(?<![0-9_])(?=[A-Z0-9])', '_', x).lower().strip() for x in browse_feat]
#### 3) 대출승인 데이터 컬럼과 대출용어 간의 차이를 확인한다
# * np.setdiff1d(ar1, ar2) : ar2 에는 없는 ar1의 고유한 값을 반환
date_feat = data.columns.values
np.setdiff1d(browse_feat, date_feat)
###### 코드 작성 부분 시작 ######
np.setdiff1d(date_feat, browse_feat)
###### 코드 작성 부분 마침 ######

#### 4) 대출 시점(대출용어 사전)에서의 용어 중 대출승인 데이터 컬럼과 같은 의미인 용어를 서로 같게 만든다
# * 대출 시점(대출용어 사전)에서의 용어 중 대출승인 데이터 컬럼과 의미가 같지만 이름이 다른 컬럼들
#         ['is_inc_v', 'mths_since_most_recent_inq','mths_since_oldest_il_open','mths_since_recent_loan_delinq', 'verified_status_joint']
# * 대출승인 데이터 컬럼에서의 이름들 (예: verified_status_join → verification_status_joint)
#         ['verification_status', 'mths_since_recent_inq', 'mo_sin_old_il_acct','mths_since_recent_bc_dlq', 'verification_status_joint']
# * np.setdiff1d / np.append 사용
wrong = ['is_inc_v', 'mths_since_most_recent_inq', 'mths_since_oldest_il_open',
         'mths_since_recent_loan_delinq', 'verified_status_joint']
correct = ['verification_status', 'mths_since_recent_inq', 'mo_sin_old_il_acct',
           'mths_since_recent_bc_dlq', 'verification_status_joint']

###### 코드 작성 부분 시작 ######
broswse_feat = np.setdiff1d(browse_feat, wrong)
###### 코드 작성 부분 마침 ######
browse_feat = np.append(browse_feat, correct)
broswse_feat

#### 5) 대출용어 사전과 대출승인 데이터 컬럼 이름이 같은 것들만 가져온다.
# * np.intersect1d(ar1, ar2) : ar1 과 ar2 의 공통된 항목들만 반환한다
###### 코드 작성 부분 시작 ######
avail_feat = np.intersect1d(browse_feat, date_feat)
###### 코드 작성 부분 마침 ######
X = data[avail_feat].copy()
X.info()
### 2. 데이터 타입별 Feature 변환
# * 결측값 처리
# * 기준정보 데이터와 대출승인 데이터의 정합성 맞추기

#### 1) 범주형 데이터 확인
# * pandas 의 select_dtypes('object') 사용

###### 코드 작성 부분 시작 ######
X.select_dtypes('object')
###### 코드 작성 부분 마침 ######
X.head()
#### 2) 범주형 문자열 데이터 중 날짜/기간/고유ID 데이터 처리
# * pandas 의 to_datetime 사용
# * .str.extract('(\d+)').astype('float') 사용
# earliest_cr_line : 대출 시작한 달
# emp_length : 근속년수
X['earliest_cr_line'] = pd.to_datetime(X['earliest_cr_line'], infer_datetime_format=True)
X['emp_length'].unique()
X['id']

X['emp_length'] = X['emp_length'].replace({'< 1 year': '0 years', '10+ years': '11 years'})
X['emp_length'] = X['emp_length'].str.extract('(\d+)').astype('float')
X['id'] = X['id'].astype('float')

### 3. 결측값 처리
# * 결측값 처리
# * 기준정보 데이터와 대출승인 데이터의 정합성 맞추기

#### 1) 컬럼별 결측값 비율 구한 후, 비율=1 인 컬럼 제거
# * .isna().mean()
# * 컬럼별 결측값 비율이 0 인 항목을 제외한 나머지 항목들을 pandas 의 sort_values() 사용하여 정렬
# * 비율=1 인 항목을 확인하여 pandas 의 drop 으로 제거

X.isna().mean()
nan_mean = X.isna().mean()
nan_mean = nan_mean[nan_mean != 0].sort_values()
nan_mean

X = X.drop(['desc', 'member_id'], axis=1, errors='ignore')
#### 2) 결측값 채우기
# * 범주형 데이터의 결측값은 공백('')으로 채운다
# * 숫자형 데이터의 결측값은 대출 데이터 특성에 따라 데이터의 최대값, 최소값으로 각각 채운다
# * 최대값으로 채우는 경우
# 'bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
# 'mths_since_last_major_derog', 'mths_since_last_record',
# 'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
# 'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
# 'pct_tl_nvr_dlq'
# * 최대값 이외에는 최소값으로 채운다

fill_empty = ['emp_title', 'verification_status_joint']

# 데이터 설명 시트의 빨간색 항목 참조
# 최대값으로 해서 해당 항목으로 인해 부실 대출 방향으로 판단하지 않도록 함
fill_max = ['bc_open_to_buy', 'mo_sin_old_il_acct', 'mths_since_last_delinq',
            'mths_since_last_major_derog', 'mths_since_last_record',
            'mths_since_rcnt_il', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
            'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
            'pct_tl_nvr_dlq']

fill_min = np.setdiff1d(X.columns.values, np.append(fill_empty, fill_max))

X[fill_empty] = X[fill_empty].fillna('')
###### 코드 작성 부분 시작 ######
X[fill_max] = X[fill_max].fillna(X[fill_max].max())
X[fill_min] = X[fill_min].fillna(X[fill_min].min())
###### 코드 작성 부분 마침 ######

### 4. 변수 간 상관성에 따른 데이터 정제
#### 1) 숫자형 컬럼 데이터의 유일한 값 개수 확인
# * pandas 의 select_dtypes('number') 사용
# * pandas 의 nunique().sort_values() 사용
# * 유일한 값이 1개인 경우/모든 행의 값이 다른 경우 둘 다 제거한다
###### 코드 작성 부분 시작 ######
num_feat = X.select_dtypes('number').columns.values
X[num_feat].nunique().sort_values()
###### 코드 작성 부분 마침 ######
X = X.drop(['num_tl_120dpd_2m', 'id'], axis=1, errors='ignore')

#### 2) 숫자형 데이터 상관도에 따른 컬럼 제거
  # * 숫자형 컬럼들 간 Pearson R 상관 계수를 구한다
  # * 상관 계수가 0.9 이상인 컬럼들 중 가장 큰 컬럼을 제거해 본다
  # * 컬럼들 간 조합 생성 : comb_num_feat = np.array(list(combinations(num_feat, 2)))
  # * Pearson R 상관 계수 구하기 : pearsonr(x1, x2)[0]
num_feat = X.select_dtypes('number').columns.values
comb_num_feat = np.array(list(combinations(num_feat, 2)))
corr_num_feat = np.array([])
for comb in comb_num_feat:
    corr = pearsonr(X[comb[0]], X[comb[1]])[0]
    corr_num_feat = np.append(corr_num_feat, corr)

###### 코드 작성 부분 시작 ######
high_corr_num = comb_num_feat[np.abs(corr_num_feat) >= 0.9]
###### 코드 작성 부분 마침 ######
high_corr_num

X = X.drop(np.unique(high_corr_num[:, 0]), axis=1, errors='ignore')

#### 3) 범주형 데이터의 유일한 값 개수 확인
  # * pandas 의 select_dtypes('object') 사용
  # * pandas 의 nunique().sort_values() 사용
  # * 유일한 값이 1개인 경우/모든 행의 값이 다른 경우 둘 다 제거한다
  # * 범주형 데이터의 경우 Encoding 시 메모리 오류를 방지하기 위해 유일한 값이 많은 경우는 제거하는 것이 좋다

cat_feat = X.select_dtypes('object').columns.values
X[cat_feat].nunique().sort_values()

X = X.drop(['url', 'emp_title'], axis=1, errors='ignore')
#### 4) 범주형 데이터 상관도에 따른 컬럼 제거
  # * 범주형 컬럼들 간 카이제곱 통계량을 사용하는 Crammer 의 V 상관 계수를 구한다
  # * Crammer 의 V 상관계수 식 구하는 방법
  # table = pd.pivot_table(X, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
  # corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1) ) )
  # * 상관 계수가 0.9 이상인 컬럼들 중 머신러닝 모델 생성 시 예측 변수의 조건으로 활용할 grade 를 제외한 후 가장 높은 컬럼을 제거한다.
  # * 컬럼들 간 조합 생성 : comb_cat_feat = np.array(list(combinations(cat_feat, 2)))

cat_feat = X.select_dtypes('object').columns.values
comb_cat_feat = np.array(list(combinations(cat_feat, 2)))
corr_cat_feat = np.array([])
for comb in comb_cat_feat:
    table = pd.pivot_table(X, values='loan_amnt', index=comb[0], columns=comb[1], aggfunc='count').fillna(0)
    corr = np.sqrt(chi2_contingency(table)[0] / (table.values.sum() * (np.min(table.shape) - 1)))
    corr_cat_feat = np.append(corr_cat_feat, corr)

high_corr_cat = comb_cat_feat[corr_cat_feat >= 0.9]
high_corr_cat
X = X.drop(np.unique(high_corr_cat[:, 1]), axis=1, errors='ignore')
### 5. 예측 변수 Feature 생성

#### 대출 상태를 보여주는 'loan_status' 를 예측 변수 Feature 로 한다
  # * loan_status 의 항목별 개수를 확인한다
  # * 건전한 상태를 나타내는 'Current, Fully Paid, In Grace Period' 를 1 로 나타낸다.
  # * 그 외는 부실한 상태를 나타내는 0 으로 나타낸다.

data['loan_status'].value_counts()
y = data['loan_status'].copy()
y = y.isin(['Current', 'Fully Paid', 'In Grace Period']).astype('int')
y.value_counts()
# 03. 모델링
#### 1) 분석 목표는 '고위험/고금리 대출 중 양호한 대출을 예측하는 것'
  # * 위험도를 나타내는 grade 컬럼에서 가장 위험한 상태인 'E' 에 해당하는 데이터만 가져온다.
  # * 위험도에 따라 설정되는 int_rate 컬럼은 제거한다.

X_mod = X[X.grade == 'E'].copy()
X_mod = X_mod.drop(['grade', 'int_rate'], axis=1, errors='ignore')
y_mod = y[X_mod.index]

#  * 훈련 데이터와 테스트 데이터를 먼저 나눈다
#  * 모델링에 사용할 훈련/검증 데이터를 그 이후에 나눈다
X_train, X_test, y_train, y_test = train_test_split(X_mod, y_mod, stratify=y_mod, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, random_state=0)
#### 2) 머신러닝 기법은 'CatBoost' 를 사용한다.
  # * CatBoost 는 범주가 많은 범주형 Feature 를 포함하는 데이터셋에 매우 효율적이다.
  # * CatBoost 는 범주형 데이터를 숫자형으로 변환하게 되고, 기본 설정으로 Mean Encoding 을 사용하는데 단순하게 평균을 사용하게 되면 Data Leakage 문제(우리가 예측해야 하는 값이 훈련 데이터의 Feature 에 들어가는 문제) 가 나타나게 되는데 이전 데이터들의 평균을 활용하는 방법을 사용하여 이를 해결해 준다
  # * Pool 을 사용하여 학습 데이터를 CatBoost 에 맞게 변환해 준다
  # * CatBoost 는 Ordered Boosting 과 Random Permutation 등의 Overfitting 을 방지하기 위한 내장 알고리즘이 있어서, 비교적 다른 Gradient Boosting 방법들에 비해 Hyper Parameter Tuning 에 자유로운 알고리즘

cat_feat_ind = (X_train.dtypes == 'object').to_numpy().nonzero()[0]
pool_train = Pool(X_train, y_train, cat_features=cat_feat_ind)
pool_val = Pool(X_val, y_val, cat_features=cat_feat_ind)
pool_test = Pool(X_test, y_test, cat_features=cat_feat_ind)

n = y_train.value_counts()
model = CatBoostClassifier(learning_rate=0.03,
                           verbose=False,
                           random_state=0)
model.fit(pool_train, eval_set=pool_val, plot=True)

#### 3) 모델의 성능
  # * Accuracy, Precision, Recall 을 사용한다
  # * Accuracy(정확도) : (실제 데이터가 예측 데이터인 수) / (전체 데이터 수) → 모델이 얼마나 정확하게 분류하는가?
  # * Precision(정밀도) : (A라고 예측한 데이터가 실제 A인 데이터 수) / (A라고 예측한 데이터 수) → 모델이 찾은 A는 얼마나 정확한가? "일반 메일을 스팸 메일로 분류해서는 안된다"
  # * Recall(재현율) : (A라고 예측한 데이터 수) / (실제 A인 데이터 수) → 모델이 얼마나 정확하게 A를 찾는가? "실제 암환자인 경우 반드시 양성으로 판단해야 한다"

y_pred_test = model.predict(pool_test)

acc_test = accuracy_score(y_test, y_pred_test)
prec_test = precision_score(y_test, y_pred_test)
rec_test = recall_score(y_test, y_pred_test)
print(f'''Accuracy (test): {acc_test:.3f} Precision (test): {prec_test:.3f} Recall (test): {rec_test:.3f}''')

cm = confusion_matrix(y_test, y_pred_test)
ax = sns.heatmap(cm, cmap='viridis_r', annot=True, fmt='d', square=True)
ax.set_xlabel('Predictive Values')
ax.set_ylabel('Actual Values')
plt.show()
## 04. Feature Importances
#### 1) Feature Importances 그래프 그리기
feat = model.feature_names_
imp = model.feature_importances_
df = pd.DataFrame({'Feature': feat, 'Importance': imp})
df = df.sort_values('Importance', ascending=False)[:10]
sns.barplot(x='Importance', y='Feature', data=df);
plt.show()

#### 2) 주요 3개 Feature 와 건전/부실 대출 간 관계
# * 대출액 히스토그램 상에서의 건전/부실 대출
good = X_mod.loc[y_mod == 1, 'loan_amnt']
bad = X_mod.loc[y_mod == 0, 'loan_amnt']

bins = 20
sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)
ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)
ax.set_ylabel('Density')
ax.legend()
plt.show()

# * 가장 최근에 대출 문의한 기간 히스토그램 상에서의 건전/부실 대출
good = X_mod.loc[y_mod == 1, 'mths_since_recent_inq']
bad = X_mod.loc[y_mod == 0, 'mths_since_recent_inq']

bins = 20
sns.distplot(good, bins=bins, label='Good loans', kde=False, norm_hist=True)
ax = sns.distplot(bad, bins=bins, label='Bad loans', kde=False, norm_hist=True)
ax.set_ylabel('Density')
ax.legend()
plt.show()

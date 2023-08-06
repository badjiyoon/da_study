import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt
import xgboost
from sklearn.preprocessing import scale
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier as xgb
import warnings

# Settings Warning and Plot Hangul
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'

# Part4. [실습8] 제조 공정 예측 유지 보수하기
# 01. 데이터 소개 및 분석프로세스 수립
# 02. 데이터 준비를 위한 EDA 및 전처리
# 0. 데이터 불러오기
telemetry = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/PdM_telemetry.csv", on_bad_lines='skip')
errors = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/PdM_errors.csv", on_bad_lines='skip')
maint = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/PdM_maint.csv", on_bad_lines='skip')
failures = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/PdM_failures.csv", on_bad_lines='skip')
machines = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/PdM_machines.csv", on_bad_lines='skip')

telemetry.head()

# 1. 각 데이터별 전처리
# telemetry
telemetry.describe()
telemetry["machineID"].nunique()
# 데이터 타입
telemetry.info()

telemetry["datetime"] = pd.to_datetime(telemetry["datetime"], format="%Y-%m-%d %H:%M:%S")
telemetry.head()

telemetry.dtypes
# 변수별 분포
# 데이터 컬럼 타입이 np.number 인 것만 가져오기
numeric_data = telemetry.select_dtypes(include=np.number)
# 데이터 컬럼 타입이 np.number인 컬럼 이름들 가져오기
l = numeric_data.columns.values
number_of_columns = 4
number_of_rows = len(l) - 1 / number_of_columns

# 컬럼별 히스토그램 그리기
for i in range(0, len(l)):
    target_data = numeric_data[l[i]]
    target_data_wo_zero = target_data[target_data > 0]
    sns.distplot(target_data_wo_zero, kde=True)  # kde : kernel density
    plt.show()

# 시계열 그래프 그려보기
plot_df = telemetry.loc[
    (telemetry["machineID"] == 1)
    & (telemetry["datetime"] > pd.to_datetime("2015-01-01"))
    & (telemetry["datetime"] > pd.to_datetime("2015-06-01")),
    ["datetime", "volt"]
]

plt.figure(figsize=(12, 6))
plt.plot(plot_df["datetime"], plot_df["volt"])
plt.title("Volt Time Series Graph")
plt.show()

# erros
errors.head()
errors.tail()
errors.info()
# 데이터 타입
errors["datetime"] = pd.to_datetime(errors["datetime"], format="%Y-%m-%d %H:%M:%S")
errors["errorID"] = errors["errorID"].astype("category")
errors.head()
# 컬럼별 개수
plt.figure(figsize=(8, 4))
errors["errorID"].value_counts().plot(kind="bar", rot=0)
plt.title("Distribution of errors data")
plt.ylabel("Count")
plt.xlabel("Error Type")
plt.show()

# maint
maint.head()
maint.tail()
maint.info()

# 데이터 타입
maint["datetime"] = pd.to_datetime(maint["datetime"], format="%Y-%m-%d %H:%M:%S")
maint["comp"] = maint["comp"].astype("category")
maint.dtypes

plt.figure(figsize=(8, 4))
maint["comp"].value_counts().plot(kind="bar", rot=0)
plt.title("Distribution of comp data")
plt.ylabel("Count")
plt.xlabel("Comp")
plt.show()

# machines
machines.head()
machines.tail()
machines.describe()

# 데이터 타입
machines["model"] = machines["model"].astype("category")
machines.dtypes

# > 모델번호별 age 분포
plt.figure(figsize=(8, 6))
_, bins, _ = plt.hist([
    machines.loc[machines["model"] == "model1", "age"],
    machines.loc[machines["model"] == "model2", "age"],
    machines.loc[machines["model"] == "model3", "age"],
    machines.loc[machines["model"] == "model4", "age"]],
    20, stacked=True, label=["model1", "model2", "model3", "model4"
                             ])
plt.title("Age Distribution by Model")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend()
plt.show()

# failures
failures.head()
failures.tail()
failures.info()

# 데이터 타입
failures["datetime"] = pd.to_datetime(failures["datetime"], format="%Y-%m-%d %H:%M:%S")
failures["failure"] = failures["failure"].astype("category")
failures.dtypes

failures.describe(include="all")
plt.figure(figsize=(8, 4))
failures["failure"].value_counts().plot(kind="bar", rot=0)
plt.title("Component Failure Distribution")
plt.ylabel("Count")
plt.xlabel("Components")
plt.show()

# 2. Feature 정제
# 일정 시간 통계량 Feature 생성
temp = []
fields = ["volt", "rotate", "pressure", "vibration"]

# 참고
index = pd.date_range('1/1/2000', periods=9, freq='T')
series = pd.Series(range(9), index=index)
series
series.resample('3T', label='right', closed='right').sum()
temp = [
    pd.pivot_table(
        telemetry,
        index="datetime",
        columns="machineID",
        values=col).resample("3H", closed="left", label="right").mean()
    for col in fields
]
temp[0].head()

temp = [
    pd.pivot_table(
        telemetry,
        index="datetime",
        columns="machineID",
        values=col).resample("3H", closed="left", label="right").mean().unstack()
    for col in fields
]
temp[0].head()

telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + "_mean_3h" for i in fields]
telemetry_mean_3h.reset_index(inplace=True)
telemetry_mean_3h.head()

temp = [
    pd.pivot_table(
        telemetry,
        index="datetime",
        columns="machineID",
        values=col).resample("3H", closed="left", label="right").std().unstack()
    for col in fields
]
temp[0].head()

telemetry_sd_3h = pd.concat(temp, axis=1)
telemetry_sd_3h.columns = [i + "_sd_3h" for i in fields]
telemetry_sd_3h.reset_index(inplace=True)
telemetry_sd_3h.head()

telemetry_feat = pd.concat([
    telemetry_mean_3h,
    telemetry_sd_3h.iloc[:, 2:6]], axis=1).dropna()
telemetry_feat.head()

telemetry_feat.describe()

# 범주형 Feature 생성
error_count = pd.get_dummies(errors)
error_count.columns = ["datetime", "machineID", "error1", "error2", "error3", "error4", "error5"]
error_count.head(15)

error_count_grouped = error_count.groupby(["machineID", "datetime"]).sum().reset_index()
error_count_grouped.head(15)

error_count_filtered = telemetry[["datetime", "machineID"]].merge(
    error_count_grouped,
    on=["machineID", "datetime"],
    how="left"
).fillna(0.0)

error_count_filtered.head()
error_count_filtered.describe()

# maint 데이터
maint.head()

comp_rep = pd.get_dummies(maint)
comp_rep.columns = ["datetime", "machineID", "comp1", "comp2", "comp3", "comp4"]
comp_rep.head()

comp_rep = comp_rep.groupby(["machineID", "datetime"]).sum().reset_index()
comp_rep.head()

comp_rep = telemetry[["datetime", "machineID"]].merge(
    comp_rep,
    on=["datetime", "machineID"],
    how="outer").fillna(0).sort_values(by=["machineID", "datetime"]
                                       )
comp_rep

components = ["comp1", "comp2", "comp3", "comp4"]
for comp in components:
    print(f"comp : {comp}")
    print(-comp_rep[comp].isnull())

for comp in components:
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), "datetime"]
    comp_rep[comp] = pd.to_datetime(comp_rep[comp].fillna(method="ffill"))  # pad, ffill : Nan 값을 앞의 값으로 채운다

comp_rep

comp_rep = comp_rep.loc[comp_rep["datetime"] > pd.to_datetime("2015-01-01")]
comp_rep.head(50)

for comp in components:
    comp_rep[comp] = (comp_rep["datetime"] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, "D")
comp_rep.head()

# 최종 Feature 정의
final_feat = telemetry_feat.merge(error_count_filtered, on=["datetime", "machineID"], how="left")
final_feat = final_feat.merge(comp_rep, on=["datetime", "machineID"], how="left")
final_feat = final_feat.merge(machines, on=["machineID"], how="left")
final_feat.head()

final_feat.describe()

# Target Feature 생성
labeled_features = final_feat.merge(failures, on=["datetime", "machineID"], how="left")

labeled_features["failure"] = labeled_features["failure"].astype(object).fillna(method="bfill", limit=7)
labeled_features["failure"] = labeled_features["failure"].fillna("none")
labeled_features["failure"] = labeled_features["failure"].astype("category")
labeled_features.head()

labeled_features["failure"].value_counts()

model_dummies = pd.get_dummies(labeled_features["model"])
labeled_features = pd.concat([labeled_features, model_dummies], axis=1)
labeled_features.drop("model", axis=1, inplace=True)
labeled_features.head()

f, ax = plt.subplots(figsize=(10, 8))
corr = labeled_features.corr(numeric_only=True)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)
plt.title("Correlation Between Variables")
plt.show()

# 모델링
# Train / Test Split
threshold_dates = [
    pd.to_datetime("2015-09-30 01:00:00"), pd.to_datetime("2015-10-01 01:00:00")
]

test_results = []
models = []
total = len(threshold_dates)

last_train_date = threshold_dates[0]
first_test_date = threshold_dates[1]

ntraining = labeled_features.loc[labeled_features["datetime"] < last_train_date]
ntesting = labeled_features.loc[labeled_features["datetime"] > first_test_date]
print(f"{ntraining.shape[0]} records for training.")
print(f"{ntesting.shape[0]} records for testing.")
print(f"{ntesting.shape[0] / ntraining.shape[0] * 100:0.1f}% of the data will be used for testing.")

# Target Feature Split
fails_train = ntraining[ntraining["failure"] != "none"].shape[0]
no_fails_train = ntraining[ntraining["failure"] == "none"].shape[0]
fails_test = ntesting[ntesting["failure"] != "none"].shape[0]
no_fails_test = ntesting[ntesting["failure"] == "none"].shape[0]

print(f"{fails_train / no_fails_train * 100:0.1f}% of the cases are training set failures.")
print(f"{fails_test / no_fails_test * 100:0.1f}% of the cases are failures in the test set.")

train_y = labeled_features.loc[labeled_features["datetime"] < last_train_date, "failure"]
train_X = labeled_features.loc[labeled_features["datetime"] < last_train_date].drop(["datetime",
                                                                                     "machineID",
                                                                                     "failure"], axis=1)
test_y = labeled_features.loc[labeled_features["datetime"] > first_test_date, "failure"]
test_X = labeled_features.loc[labeled_features["datetime"] > first_test_date].drop(["datetime",
                                                                                    "machineID",
                                                                                    "failure"], axis=1)
# XGBoost Model
# Fitting
model = xgb(n_jobs=-1)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
train_y = le.fit_transform(train_y)
model.fit(train_X, train_y)
# Prediction
test_result = pd.DataFrame(labeled_features.loc[labeled_features["datetime"] > first_test_date])
test_result["predicted_failure"] = model.predict(test_X)
test_results.append(test_result)
models.append(model)

# Feature Importance
plt.figure(figsize=(10, 10))
labels, importances = zip(
    *sorted(zip(test_X.columns, models[0].feature_importances_), reverse=False, key=lambda x: x[1]))
plt.yticks(range(len(labels)), labels)
_, labels = plt.xticks()
plt.setp(labels, rotation=0)
plt.barh(range(len(importances)), importances)
plt.ylabel("features")
plt.xlabel("Importance (%)")
plt.title("Importance of Characteristics According to the Model")
plt.show()


# 모델 평가함수 생성
def Evaluate(predicted, actual, labels):
    output_labels = []
    output = []
    predicted = le.inverse_transform(predicted)

    # confusion matrix
    cm = confusion_matrix(actual, predicted, labels=labels)

    # precision, recall, and F1 score
    accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
    precision = precision_score(actual, predicted, average=None, labels=labels)
    recall = recall_score(actual, predicted, average=None, labels=labels)
    f1 = 2 * precision * recall / (precision + recall)
    output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
    output_labels.extend(["accuracy", "precision", "recall", "F1"])

    output_df = pd.DataFrame(output, columns=labels)
    output_df.index = output_labels

    return output_df


evaluation_results = []
test_result = test_results[0]
test_result["failure"]

"""#### Confusion Matrix 평가"""

evaluation_result = Evaluate(actual=test_result["failure"],
                             predicted=test_result["predicted_failure"],
                             labels=["none", "comp1", "comp2", "comp3", "comp4"])

skplt.metrics.plot_confusion_matrix(
    test_result["failure"],
    le.inverse_transform(test_result["predicted_failure"]),
    normalize=False,
    title="Confusion Matrix"
)

skplt.metrics.plot_confusion_matrix(
    test_result["failure"],
    le.inverse_transform(test_result["predicted_failure"]),
    normalize=True,
    title="Normalized Confusion Matrix",
)
plt.show()

evaluation_results.append(evaluation_result)
evaluation_results[0]

#### Accuracy-Sensitivity 곡선
# * 재현율 : 실제 Positive 인 데이터 예측을 Negative 로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우 중요
# * 정밀도 : 실제 Negative 인 데이터 예측을 Positive 로 잘못 판단하게 되면 업무상 큰 영향이 발생하는 경우 중요

skplt.metrics.plot_precision_recall_curve(
    test_y,
    model.predict_proba(test_X),
    title="Accuracy-Sensitivity Curve",
    figsize=(10,10)
)
plt.show()

recall_df = pd.DataFrame([evaluation_results[0].loc["recall"].values],
                         columns=["none", "comp1", "comp2", "comp3", "comp4"],
                         index=["Component Sensitivity"])
recall_df.T

# 예측 테스트
test_values = train_X.iloc[0].values
test_values

single_test = pd.DataFrame([test_values], columns=test_X.columns, index=[0])
single_test

probas = model.predict_proba(single_test)
prediction = model.predict(single_test)
ordered_classes = np.unique(np.array(test_y))

results = pd.DataFrame(probas,
                       columns=ordered_classes,
                       index=[0])
print(f"Prediction : {prediction[0]}")
results

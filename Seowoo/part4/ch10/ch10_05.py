# 뉴스 분류하기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)
## 1. Data

### 1.1 Data Load
from sklearn.datasets import fetch_20newsgroups

newsgroup = fetch_20newsgroups()
data, target = newsgroup["data"], newsgroup["target"]
print(data[0])
target[0]
newsgroup["target_names"]
### 1.2 Data Split
# 아래의 뉴스 그룹만 사용
# - 'talk.politics.guns'
# - 'talk.politics.mideast'
# - 'talk.politics.misc'
# - 'talk.religion.misc
len(newsgroup["target_names"])
text = pd.Series(data, name="text")
target = pd.Series(target, name="target")

df = pd.concat([text, target],axis=1)
print("df is here")
print(df)


df.target.value_counts().sort_index()
df.query("16 <= target <= 19")
df_sample = df.query("16 <= target <= 19")

data = df_sample.text
target = df_sample.target

np.array(data).shape
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.7, random_state=2021
)
### 1.2 Count Vectorize
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('punkt')
# 뉴스에 모두 등장한 단어를 사용
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_data)
len(cnt_vectorizer.vocabulary_)
# 최소 10개의 뉴스에서 등장한 단어를 사용
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize, min_df=10)
cnt_vectorizer.fit(train_data)
len(cnt_vectorizer.vocabulary_)
train_matrix = cnt_vectorizer.transform(train_data)
test_matrix = cnt_vectorizer.transform(test_data)
## 2. XGBoost
import xgboost as xgb

xgb_clf = xgb.XGBClassifier()
### 2.1 학습
print("issue is occured")
print(train_matrix)
print(train_target)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_target = le.fit_transform(train_target)

xgb_clf.fit(train_matrix, train_target)
### 2.2 예측
xgb_train_pred = xgb_clf.predict(train_matrix)
xgb_test_pred = xgb_clf.predict(test_matrix)

### 2.3 평가
from sklearn.metrics import accuracy_score

xgb_train_acc = accuracy_score(train_target, xgb_train_pred)
xgb_test_acc = accuracy_score(test_target, xgb_test_pred)
print(f"XGBoost Train accuracy is {xgb_train_acc:.4f}")
print(f"XGBoost Test accuracy is {xgb_test_acc:.4f}")


## 3. Light GBM
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier()

### 3.1 학습
train_matrix
train_matrix.toarray()
lgb_clf.fit(train_matrix.toarray(), train_target)

### 3.2 예측
lgb_train_pred = lgb_clf.predict(train_matrix.toarray())
lgb_test_pred = lgb_clf.predict(test_matrix.toarray())

### 3.3 평가
lgb_train_acc = accuracy_score(train_target, lgb_train_pred)
lgb_test_acc = accuracy_score(test_target, lgb_test_pred)
print(f"Light Boost train accuracy is {lgb_train_acc:.4f}")
print(f"Light Boost test accuracy is {lgb_test_acc:.4f}")


## 4. CatBoost
import catboost as cb

cb_clf = cb.CatBoostClassifier()

### 4.1 학습
cb_clf.fit(train_matrix, train_target, verbose=False)

### 4.2 예측
cb_train_pred = cb_clf.predict(train_matrix)
cb_test_pred = cb_clf.predict(test_matrix)

### 4.3 평가
cb_train_acc = accuracy_score(train_target, cb_train_pred)
cb_test_acc = accuracy_score(test_target, cb_test_pred)
print(f"Cat Boost train accuracy is {cb_train_acc:.4f}")
print(f"Cat Boost test accuracy is {cb_test_acc:.4f}")


## 5. 마무리
print(f"XGBoost test accuray is {xgb_test_acc:.4f}")
print(f"Light Boost test accuray is {lgb_test_acc:.4f}")
print(f"Cat Boost test accuray is {cb_test_acc:.4f}")
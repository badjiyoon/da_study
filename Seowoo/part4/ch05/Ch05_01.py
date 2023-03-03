# 스팸 문자를 Naive Bayes를 이용해 분류하기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)
## 1. Data
### 1.1 Data Load
# `sms_spam.csv` 데이터는 문자 내용이 스팸인지 아닌지를 구분하기 위한 데이터 입니다.
spam = pd.read_csv("sms_spam.csv")
text = spam["text"]
label = spam["type"]
### 1.2 Data EDA
text[0]
label[0]
label.value_counts()
### 1.3 Data Cleaning
# 정답의 문자를 숫자로 변환시켜줍니다.
# ham은 0으로, spam은 1로 변환 시켜주겠습니다.
label = label.map({"ham": 0, "spam": 1})
label.value_counts()
# text를 문자만 존재하도록 정리해줍니다.
# regex를 통해 영어, 숫자 그리고 띄어쓰기를 제외한 모든 단어를 지우도록 하겠습니다.
re_pattern = "[^a-zA-Z0-9\ ]"
text[0]
text.iloc[:1].str.replace(re_pattern, "", regex=True)[0]
text = text.str.replace(re_pattern, "", regex=True)
# 그리고 나서 대문자들을 모두 소문자로 바꿔 줍니다.
text[0]
text.iloc[:1].str.lower()[0]
text = text.str.lower()
text[0]
### 1.4 Data Split
from sklearn.model_selection import train_test_split

train_text, test_text, train_label, test_label = train_test_split(
    text, label, train_size=0.7, random_state=2021
)
print(f"train_data size: {len(train_label)}, {len(train_label)/len(text):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(text):.2f}")
## 2. Count Vectorize
# 이제 Naive Bayes를 학습시키기 위해서 각 문장에서 단어들이 몇 번 나왔는지로 변환해야 합니다.
### 2.1 word tokenize

# 문장을 단어로 나누는 데에는 `nltk` 패키지의 `word_tokenize`를 이용합니다.
import nltk
from nltk import word_tokenize

nltk.download('punkt')
train_text.iloc[0]
word_tokenize(train_text.iloc[0])
### 2.2 count vectorize

# 다음은 `sklearn.feature_extraction.text`의 `CountVectorizer`를 이용해 단어들을 count vector로 만들어 보겠습니다.
from sklearn.feature_extraction.text import CountVectorizer
# 우선 예시로 2개의 문장으로 CountVectorizer를 학습해 보겠습니다.
train_text.iloc[:2].values
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text.iloc[:2])
# 문장에서 나온 단어들은 다음과 같습니다.
cnt_vectorizer.vocabulary_
vocab = sorted(cnt_vectorizer.vocabulary_.items(), key=lambda x: x[1])
vocab = list(map(lambda x: x[0], vocab))
vocab
sample_cnt_vector = cnt_vectorizer.transform(train_text.iloc[:2]).toarray()
sample_cnt_vector
train_text.iloc[:2].values
pd.DataFrame(sample_cnt_vector, columns=vocab)
#### 2.2.1 학습
# 이제 모든 데이터에 대해서 진행하겠습니다.
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text)
# 전체 단어는 7908개가 존재합니다.
len(cnt_vectorizer.vocabulary_)
#### 2.2.2 예측
train_matrix = cnt_vectorizer.transform(train_text)
test_matrix = cnt_vectorizer.transform(test_text)
# 만약 존재하지 않는 단어가 들어올 경우 어떻게 될까요?
# CountVectorize는 학습한 단어장에 존재하지 않는 단어가 들어오게 될 경우 무시합니다.
cnt_vectorizer.transform(["notavailblewordforcnt"]).toarray().sum()
## 3. Naive Bayes
# 분류를 위한 Naive Bayes 모델은 `sklearn.naive_bayes`의 `BernoulliNB`를 사용하면 됩니다.
from sklearn.naive_bayes import BernoulliNB

naive_bayes = BernoulliNB()
### 3.1 학습
naive_bayes.fit(train_matrix, train_label)
### 3.2 예측
train_pred = naive_bayes.predict(train_matrix)
test_pred = naive_bayes.predict(test_matrix)
### 3.3 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_label, train_pred)
test_acc = accuracy_score(test_label, test_pred)
print(f"Train Accuracy is {train_acc:.4f}")
print(f"Test Accuracy is {test_acc:.4f}")

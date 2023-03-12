#CH05_01. 스팸 문자를 Naive Bayes를 이용해 분류하기

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load

spam = pd.read_csv("C:/Users/sewon/Documents/da_study/Sewon/part4/ch5/sms_spam.csv")
#sms_spam.csv 데이터는 문자 내용이 스팸인지 아닌지를 구분하기 위한 데이터

"""
*파일을 불러올 때, 파일 경로를 전체 다 써주면 됨됨
그냥 파일 경로를 쓰면 \을 unicode로 인식해서 오류 발생

\ -> /로 바꿔주거나
\ -> \\로 바꿔서 경로 입력해주면 됨
"""

text = spam["text"]
label = spam["type"]

#1.2 Data EDA
text[0]
label[0]

label.value_counts()

#1.3 Data Cleaning
"""
정답의 문자를 숫자로 변환
ham은 0으로, spam은 1로 변환 시켜
"""

label = label.map({"ham": 0, "spam": 1})
label.value_counts()

"""
text를 문자만 존재하도록 정리
regex를 통해 영어, 숫자 그리고 띄어쓰기를 제외한 모든 단어 삭제
"""

re_pattern = "[^a-zA-Z0-9\ ]" # ^ 지우겠다는 의미, regular expression
text[0]
text.iloc[:1].str.replace(re_pattern, "", regex=True)[0]
text = text.str.replace(re_pattern, "", regex=True)

"""
그리고 나서 대문자들을 모두 소문자로 바꿔 줍니다.
"""

text[0]
text.iloc[:1].str.lower()[0]
text = text.str.lower()
text[0]

#1.4 Data Split

from sklearn.model_selection import train_test_split

train_text, test_text, train_label, test_label = train_test_split(
    text, label, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_label)}, {len(train_label)/len(text):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(text):.2f}")

#2. Count Vectorize
#Naive Bayes를 학습시키기 위해서 각 문장에서 단어들이 몇 번 나왔는지로 변환

#2.1 word tokenize
#문장을 단어로 나누는 데에는 nltk 패키지의 word_tokenize를 이용

import nltk
from nltk import word_tokenize

nltk.download('punkt')

train_text.iloc[0]
word_tokenize(train_text.iloc[0]) #띄어쓰기 단위로 나누기

#2.2 count vectorize
"""
다음은 sklearn.feature_extraction.text의 CountVectorizer를 이용
단어들을 count vector로 생성
"""

from sklearn.feature_extraction.text import CountVectorizer

train_text.iloc[:2].values #2개의 문장으로 CountVectorizer를 학습

cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text.iloc[:2])

cnt_vectorizer.vocabulary_ #단어가 몇 번씩 나왔는지 확인

vocab = sorted(cnt_vectorizer.vocabulary_.items(), key=lambda x: x[1])
#x[1]에 있는 단어 등장 횟수를 기준으로 오름차순 정렬
vocab

vocab = list(map(lambda x: x[0], vocab))
#lambda x: for문의 기능과 비슷, x[0]에 있는 단어들만 남기기
vocab

sample_cnt_vector = cnt_vectorizer.transform(train_text.iloc[:2]).toarray()
sample_cnt_vector

train_text.iloc[:2].values

pd.DataFrame(sample_cnt_vector, columns=vocab)

#2.2.1 학습
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text)

len(cnt_vectorizer.vocabulary_) #7908

train_matrix = cnt_vectorizer.transform(train_text)
test_matrix = cnt_vectorizer.transform(test_text)

cnt_vectorizer.transform(["notavailblewordforcnt"]).toarray().sum()

#3. Naive Bayes
"""
분류를 위한 Naive Bayes 모델은 
sklearn.naive_bayes의 BernoulliNB를 사용
"""

from sklearn.naive_bayes import BernoulliNB

naive_bayes = BernoulliNB()

#3.1 학습
naive_bayes.fit(train_matrix, train_label)

#3.2 예측
train_pred = naive_bayes.predict(train_matrix)
test_pred = naive_bayes.predict(test_matrix)

#3.3 평가
from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_label, train_pred)
test_acc = accuracy_score(test_label, test_pred)

print(f"Train Accuracy is {train_acc:.4f}")
print(f"Test Accuracy is {test_acc:.4f}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(2021)

# 스팸 문자 나이브베이즈를 이용해 분류하기

# 1. Data
# 1.1 Data Load
spam = pd.read_csv("/Users/jiyoonkim/Documents/da_study/comFiles/sms_spam.csv")
text = spam["text"]
label = spam["type"]

# 1.2 Data EDA
text[0]
label[0]

# 1.3 Data Cleaning
# 정답의 문자를 숫자로 변환
# ham은 0으로, spam은 1로 변환
label = label.map({"ham": 0, "spam": 1})
# text를 문자만 존재하도록 정리
# regex를 통해 영어, 숫자, 그리고 띄어쓰기를 제외한 모든 단어 제거
re_pattern = "[^a-zA-Z0-9\ ]"
text[0]
text.iloc[:1].str.replace(re_pattern, "", regex=True)[0]
text = text.str.replace(re_pattern, "", regex=True)
# 그리고 나서 대문자를 모두 소문자로 변경
text[0]
text.iloc[:1].str.lower()[0]
text = text.str.lower()
text[0]
text = text.str.replace(re_pattern, "", regex=True)
# 1.4 Data Split
from sklearn.model_selection import train_test_split

train_text, test_text, train_label, test_label = train_test_split(
    text, label, train_size=0.7, random_state=2021
)

print(f"train_data size: {len(train_label)}, {len(train_label)/len(text):.2f}")
print(f"test_data size: {len(test_label)}, {len(test_label)/len(text):.2f}")

# 2. Count Vectorize
# 2.1 word tokenize
# 문장을 단어로 나누는데에는 nltk 패키지의 word_tokenize를 이용
import nltk
from nltk import word_tokenize
nltk.download("punkt")
train_text.iloc[0]
word_tokenize(train_text.iloc[0])

# 2.2 count vectorize
from sklearn.feature_extraction.text import CountVectorizer
train_text.iloc[:2].values
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text.iloc[:2])
cnt_vectorizer.vocabulary_

vocab = sorted(cnt_vectorizer.vocabulary_.items(), key=lambda x: x[1])
vocab = list(map(lambda x: x[0], vocab))
vocab
sample_cnt_vector = cnt_vectorizer.transform(train_text.iloc[:2]).toarray()
sample_cnt_vector
train_text.iloc[:2].values
pd.DataFrame(sample_cnt_vector, columns=vocab)

# 이제 모든 데이터에 대해서 진행하겠습니다.
cnt_vectorizer = CountVectorizer(tokenizer=word_tokenize)
cnt_vectorizer.fit(train_text)
len(cnt_vectorizer.vocabulary_)
train_matrix = cnt_vectorizer.transform(train_text)
test_matrix = cnt_vectorizer.transform(test_text)
cnt_vectorizer.transform(["notavailblewordforcnt"]).toarray().sum()
# 3. Naive Bayes
from sklearn.naive_bayes import BernoulliNB
naive_bayes = BernoulliNB()

naive_bayes.fit(train_matrix, train_label)

train_pred = naive_bayes.predict(train_matrix)
test_pred = naive_bayes.predict(test_matrix)

from sklearn.metrics import accuracy_score

train_acc = accuracy_score(train_label, train_pred)
test_acc = accuracy_score(test_label, test_pred)

print(f"Train Accuracy is {train_acc:.4f}")
print(f"Test Accuracy is {test_acc:.4f}")

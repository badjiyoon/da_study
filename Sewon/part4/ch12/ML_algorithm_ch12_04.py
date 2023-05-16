#CH12_04. TF-IDF 실습
"영화 줄거리 데이터에 TF-IDF를 적용해 영화별 유사도를 계산"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

#1. Data
#1.1 Data Load
"데이터에서 영화 제목을 나타내는 `title`과 줄거리 `overview` 컬럼을 이용"

df = pd.read_csv("C:/Users/sewon/Documents/da_study/Sewon/part4/ch12/movies_metadata.csv")
df = df[["title", "overview"]]

df = df.iloc[:1000] #전체 데이터의 1,000개만 사용
df.shape
df

#1.2 Data Cleaning
"overview'가 결측값인 경우 빈 str으로 대체"

df["overview"].isna().sum()
df["overview"] = df["overview"].fillna('')

#2. TF-IDF 계산
"sklearn.feature_extraction.text`의 `TfidfVectorizer`을 이용해 TF-IDF 결과 값을 계산"

from sklearn.feature_extraction.text import TfidfVectorizer

#2.1 Sample Data
df["overview"].values[:2]

transformer = TfidfVectorizer(stop_words='english')
tfidf_matrix = transformer.fit_transform(df['overview'].values[:2])

tfidf_matrix.toarray()

transformer.get_feature_names()[:10]

pd.DataFrame(tfidf_matrix.toarray(), columns=transformer.get_feature_names()).T.head(10)

#2.2 학습
transformer = TfidfVectorizer(stop_words='english')

# 2.3 변환
tfidf_matrix = transformer.fit_transform(df['overview'])
tfidf_matrix.toarray()

transformer.get_feature_names()[-5:] #키워드 확인

#3. 영화별 유사도 계산
"코사인 유사도를 이용해 영화별 유사도를 계산"

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)
similarity

#4. 유사한 영화 추천
idx = 998 #데이터 인덱스 998과 유사한 영화를 추천
print(df.loc[idx, 'title'])

similarity_one_idx = similarity[idx]
#similarity에서 998번째 영화와 다른 영화 사이의 유사도를 추출, 유사도 높은 인덱스를 반환

"""
1) argsort는 값을 오름차순으로 정렬할때 해당하는 인덱스를 반환
2) argsort에 역순을 취해 가장 유사한 인덱스가 앞으로 오도록 정렬

"""
order_idx = similarity_one_idx.argsort()[::-1]
order_idx[:100]

"결과 자기 자신과의 유사도가 가장 높고 이후 유사한 영화의 인덱스 추출"
top5 = order_idx[:6]
top5

"""
기존 데이터에서 각 인덱스에 해당하는 영화의 제목은 다음과 같습니다.
"Robin Hood: Prince of Thieves"와 유사한 "Robin Hood" 영화가 추천되는 것을 확인
"""
df.loc[top5, 'title']

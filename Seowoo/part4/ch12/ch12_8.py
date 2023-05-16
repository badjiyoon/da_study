# KNN 협업 필터링 실습
# 유저-영화 평점 데이터를 이용해 유저가 아직 평가하지 않은 영화를 추천을 해보겠습니다.
import os

import numpy as np
import pandas as pd

rootPath = os.getcwd() + "/Seowoo/part4/ch12"


np.random.seed(2021)
## 1. Data
### 1.1 Data Load
# 데이터에서 유저 고유 아이디를 나타내는 `userId`, 영화 고유 아이디를 나타내는 `movieId`, 유저가 영화를 평가한 점수 `rating` 컬럼을 이용합니다.

ratings = pd.read_csv(rootPath + "/ratings_small.csv")
ratings = ratings[["userId", "movieId", "rating"]]
ratings.head()

# 다른 두 데이터를 이용해 `ratings` 데이터의 `movieId`에 맞는 영화 제목을 얻습니다.
movies = pd.read_csv(rootPath + "/movies_metadata.csv", low_memory=False)
links = pd.read_csv(rootPath + "/links_small.csv", low_memory=False)


### 1.2 Data Preprocessing
# `movies` 데이터에서 "tt숫자"로 이루어진 `imdb_id`에서 숫자 부분과
# `links` 데이터의 "숫자"로 이루어진 `imdbId`와 연결합니다.
movies = movies.fillna('')
movies = movies[movies["imdb_id"].str.startswith('tt')]
movies["imdbId"] = movies["imdb_id"].apply(lambda x: int(x[2:]))
movies = movies.merge(links, on="imdbId")
movies = movies[["title", "movieId"]]
movies = movies.set_index("movieId")
movies.head()

# `pivot`함수를 이용해 유저 아이디가 인덱스이고, 영화 아이디가 컬럼, 값이 평가 점수인 `user_movie_matrix`를 만듭니다.
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating",
)

user_movie_matrix.iloc[-5:, -5:]

# 유저가 평가하지 않은 영화에 대해서 결측값을 0으로 대체합니다.
user_movie_matrix = user_movie_matrix.fillna(0)
user_movie_matrix.shape
user_movie_matrix


## 2. KNN Basic
# k가 5인 KNN Basic을 이용해 유저 "124" 가 아직 평가하지 않은 영화 "648"에 대한 점수를 예측해 보겠습니다.
k = 5
user_i = 124
movie_id = 648

user_movie_matrix[movie_id].loc[user_i]


### 2.1 유저 간의 유사도를 계산한다.

# `cosin_similarity` 함수를 이용해 유저별 코사인 유사도를 계산합니다.
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user_movie_matrix)
user_similarity.shape
user_similarity[:10, :10]
user_similarity = pd.DataFrame(
    data=user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index,
)
user_similarity.head(5)

### 2.2 아이템 i를 평가한 유저들 중에서 유저 u와 비슷한 유저 k명을 찾는다.
# 이제 유저 "124"와 유사한 다른 유저 k명을 찾습니다.
user_i_similarity = user_similarity.loc[user_i]
user_i_similarity
user_i_similarity = user_i_similarity.sort_values(ascending=False)
user_i_similarity

# 유사도 상위 k명의 유사도와 id 추출합니다.
# 이때 가장 유사도가 높은 id는 user_i로 제외합니다.
top_k_similarity = user_i_similarity[1: k + 1]
top_k_similar_user_ids = top_k_similarity.index
top_k_similar_user_ids
top_k_similarity

### 2.3 K명의 유사한 유저들이 아이템 i에 평가한 선호도를 유사도 기준으로 가중 평균한다.
tok_k_similar_ratings = user_movie_matrix.loc[top_k_similar_user_ids, movie_id]
tok_k_weighted_ratings = tok_k_similar_ratings * top_k_similarity
movie_id
tok_k_similar_ratings

# 평가 점수가 있는 유저에 대한 Weight 추출합니다
top_k_weight = (tok_k_similar_ratings > 0) * top_k_similarity
top_k_weight

# 유사도가 곱해진 평가 점수의 합을 유사도 합으로 나눕니다.
weighted_rating = tok_k_weighted_ratings.sum()
weight = top_k_weight.sum()
weight

# weight가 0보다 작은 경우 유저 모두 평가하지 않은 경우입니다.
if weight > 0:
    prediction_rating = weighted_rating / weight
else:
    prediction_rating = 0
prediction_rating


### 2.4 예측 선호도가 높은 아이템을 유저에게 추천한다.


# 모든 영화에 대해서 점수를 예측하고 예측 평가 점수가 높은 영화를 유저에게 추천합니다.
#### 2.4.1 선호도 계산
prediction_dict = {}

# 모든 영화 아이디에 대해 평점 예측
for movie_id in user_movie_matrix.columns:

    # 이미 유저가 평가한 경우 제외
    if user_movie_matrix.loc[user_i, movie_id] > 0:
        continue

    tok_k_similar_ratings = user_movie_matrix.loc[top_k_similar_user_ids, movie_id]

    top_k_weighted_ratings = tok_k_similar_ratings * top_k_similarity
    top_k_weight = (tok_k_similar_ratings > 0) * top_k_similarity

    weighted_rating = top_k_weighted_ratings.sum()
    weight = top_k_weight.sum()

    if weight > 0:
        prediction_rating = weighted_rating / weight
    else:
        prediction_rating = 0

    # 영화 아이디별로 예측 평가 점수 저장
    prediction_dict[movie_id] = prediction_rating

# 영화 아이디별 예측 평가 점수를 내림차순으로 정렬합니다.
prediction = pd.Series(prediction_dict).sort_values(ascending=False)


#### 2.4.2 상위 아이템 추출
# 예측 평가 점수 상위 10개의 영화 아이디 추출합니다.
recommend = prediction[:10].index
movies.loc[recommend]


## 3. KNN with Means
# k가 5인 KNN Basic을 이용해 유저 "124" 가 아직 평가하지 않은 영화 "31"에 대한 점수를 예측하는 과정입니다.
user_id = 124
k = 5
movie_i = 648

# pivot함수를 이용해 영화 아이디가 인덱스이고, 유저 아이디가 컬럼, 값이 평가 점수인 movie_user_matrix를 만듭니다.

# 결측값은 0으로 대체합니다.
movie_user_matrix = ratings.pivot(
    index="movieId",
    columns="userId",
    values="rating",
)
movie_user_matrix = movie_user_matrix.fillna(0)


### 3.1 아이템간의 유사도를 계산한다.

# 영화간의 피어슨 유사도를 계산합니다.
movie_similarity = np.corrcoef(movie_user_matrix)
movie_similarity = pd.DataFrame(
    data=movie_similarity,
    index=movie_user_matrix.index,
    columns=movie_user_matrix.index,
)
movie_similarity.shape


### 3.2 아이템 i와 비슷한 아이템을 k개 찾는다.

# 영화 "648"과 유사한 다른 영화 k개를 찾습니다.
# 우선 movie_i와 다른 영화 간의 유사도 추출
movie_i_similarity = movie_similarity.loc[movie_i]

# 다른 영화와의 유사도 내림차순 정렬합니다.
movie_i_similarity = movie_i_similarity.sort_values(ascending=False)

# 유사도 상위 k개의 유사도와 id 추출 합니다.
# 이 때 가장 유사도가 높은 id는 movie_i로 제외합니다.
top_k_similarity = movie_i_similarity[1: k + 1]
top_k_similar_movie_ids = top_k_similarity.index
top_k_similar_movie_ids
top_k_similarity


### 3.3 아이템 i의 평균 선호도를 계산한다.

# 영화별로 특징이 되는 평균 선호도를 계산합니다.
# 평점이 0인 경우 평가하지 않음을 반영하기 위해 결측값으로 대체합니다.
movie_user_matrix = movie_user_matrix.replace(0, np.NaN)
movie_bias = movie_user_matrix.mean(1)
movie_bias


### 3.4 유저가 평가한 K개의 아이템의 선호도의 편차를 유사도 기준으로 가중 평균한다.
#### 3.4.1 유저별 영화 평가 점수 편차 계산
movie_user_matrix_wo_bias = movie_user_matrix.sub(movie_bias, axis=0)
movie_user_matrix_wo_bias


#### 3.4.2 상위 k개의 선호도 추출
tok_k_similar_ratings = movie_user_matrix_wo_bias.loc[top_k_similar_movie_ids, user_id]
top_k_weighted_ratings = tok_k_similar_ratings * top_k_similarity
tok_k_similar_ratings
top_k_weighted_ratings

# 추출된 영화중 평가 점수가 있는 영화에 대한 가중치만 남깁니다.
top_k_weight = (pd.notna(tok_k_similar_ratings)) * top_k_similarity
top_k_weight


#### 3.4.3 가중 평균
# 유사도가 곱해진 평가 점수의 편차 합을 유사도 합으로 나눔
weighted_rating = top_k_weighted_ratings.sum()
weight = top_k_weight.sum()
weight

# 영화 평균 평점 추출
bias = movie_bias.loc[movie_i]
bias
if weight != 0:
    # 평균 평점에 가중 편차 합
    prediction_rating = bias + weighted_rating / weight

# weight가 0인 경우 유사 영화 모두 평가하지 않은 경우
else:
    prediction_rating = 0
prediction_rating
### 3.5 예측 선호도가 높은 아이템을 유저에게 추천한다.

# 모든 영화에 대해서 점수를 예측하고 예측 평가 점수가 높은 영화를 유저에게 추천합니다.
prediction_dict = {}

# 모든 영화 아이디에 대해 평점 예측
for movie_id in movie_user_matrix.index:

    # 이미 유저가 평가한 경우 제외
    if movie_user_matrix.loc[movie_i, user_id] > 0:
        continue

    tok_k_similar_ratings = movie_user_matrix_wo_bias.loc[top_k_similar_movie_ids, user_id]

    top_k_weighted_ratings = tok_k_similar_ratings * top_k_similarity
    top_k_weight = (tok_k_similar_ratings != 0) * top_k_similarity

    weighted_rating = top_k_weighted_ratings.sum()
    weight = top_k_weight.sum()

    bias = movie_bias.loc[movie_i]

    if weight > 0:
        prediction_rating = bias + weighted_rating / weight
    else:
        prediction_rating = 0

    # 영화 아이디 별로 예측 평가 점수 저장
    prediction_dict[movie_id] = prediction_rating

# 영화 아이디별 예측 평가 점수를 내림차순으로 정렬
prediction = pd.Series(prediction_dict).sort_values(ascending=False)

# 예측 평가 점수 상위 10개의 영화 아이디 추출
recommend = prediction[:10].index
movies.loc[recommend]

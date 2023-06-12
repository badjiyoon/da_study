# KNN 협업 필터링 실습
import pandas as pd
import numpy as np

np.random.seed(2021)
## 1. Data
### 1.1 Data Load
ratings = pd.read_csv("ratings_small.csv")
ratings = ratings[["userId", "movieId", "rating"]]
ratings.head()
movies = pd.read_csv("movies_metadata.csv")
links = pd.read_csv("links_small.csv")
### 1.2 Data Preprocessing
movies = movies.fillna('')
movies = movies[movies["imdb_id"].str.startswith('tt')]
movies["imdbId"] = movies["imdb_id"].apply(lambda x: int(x[2:]))
movies = movies.merge(links, on="imdbId")
movies = movies[["title", "movieId"]]
movies = movies.set_index("movieId")
movies.head()
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating",
)
user_movie_matrix.iloc[-5:, -5:]
user_movie_matrix = user_movie_matrix.fillna(0)
user_movie_matrix.shape
## 2. KNN Basic
k = 5
user_i = 124
movie_id = 648
### 2.1 유저 간의 유사도를 계산한다.

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
user_i_similarity = user_similarity.loc[user_i]
user_i_similarity
user_i_similarity = user_i_similarity.sort_values(ascending=False)
user_i_similarity
top_k_similarity = user_i_similarity[1: k + 1]
top_k_similar_user_ids = top_k_similarity.index
top_k_similar_user_ids
top_k_similarity
### 2.3 K명의 유사한 유저들이 아이템 i에 평가한 선호도를 유사도 기준으로 가중 평균한다.
tok_k_similar_ratings = user_movie_matrix.loc[top_k_similar_user_ids, movie_id]
movie_id
tok_k_similar_ratings
top_k_weight = (tok_k_similar_ratings > 0) * top_k_similarity
top_k_weight
weighted_rating = top_k_weighted_ratings.sum()
weight = top_k_weight.sum()
weight
if weight > 0:
    prediction_rating = weighted_rating / weight
else:
    prediction_rating = 0
prediction_rating
### 2.4 예측 선호도가 높은 아이템을 유저에게 추천한다.


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
prediction = pd.Series(prediction_dict).sort_values(ascending=False)
#### 2.4.2 상위 아이템 추출
recommend = prediction[:10].index
movies.loc[recommend]
## 3. KNN with Means
user_id = 124
k = 5
movie_i = 648

movie_user_matrix = ratings.pivot(
    index="movieId",
    columns="userId",
    values="rating",
)
movie_user_matrix = movie_user_matrix.fillna(0)
### 3.1 아이템간의 유사도를 계산한다.

movie_similarity = np.corrcoef(movie_user_matrix)
movie_similarity = pd.DataFrame(
    data=movie_similarity,
    index=movie_user_matrix.index,
    columns=movie_user_matrix.index,
)
movie_similarity.shape
### 3.2 아이템 i와 비슷한 아이템을 k개 찾는다.

movie_i_similarity = movie_similarity.loc[movie_i]
movie_i_similarity = movie_i_similarity.sort_values(ascending=False)
top_k_similarity = movie_i_similarity[1: k + 1]
top_k_similar_movie_ids = top_k_similarity.index
top_k_similar_movie_ids
top_k_similarity
### 3.3 아이템 i의 평균 선호도를 계산한다.

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
top_k_weight = (pd.notna(tok_k_similar_ratings)) * top_k_similarity
top_k_weight
#### 3.4.3 가중 평균
weighted_rating = top_k_weighted_ratings.sum()
weight = top_k_weight.sum()
weight
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
prediction = pd.Series(prediction_dict).sort_values(ascending=False)
recommend = prediction[:10].index
movies.loc[recommend]
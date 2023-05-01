# Matrix Factorization 실습
# KNN과 동일한 `ratings` 데이터에 모델 기반 협업필터링 방법 중 하나인 Matrix Factorization을 적용합니다.
import pandas as pd
import numpy as np


np.random.seed(2021)
## 1. Data
### 1.1 Data Load
# 유저-영화 평점 데이터를 이용해 유저가 아직 평가하지 않은 영화를 추천하겠습니다.

# 데이터에서 유저 고유 아이디를 나타내는 `userId`, 영화 고유 아이디를 나타내는 `movieId`, 유저가 영화를 평가한 점수 `rating` 컬럼을 이용합니다.
ratings = pd.read_csv("ratings_small.csv")
ratings = ratings[["userId", "movieId", "rating"]]
ratings.head()
# 다른 두 데이터를 이용해 `ratings` 데이터의 `movieId`에 맞는 영화 제목을 얻습니다.
movies = pd.read_csv("movies_metadata.csv")
links = pd.read_csv("links_small.csv")
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

# 결측값은 0으로 대체합니다.
user_movie_matrix = ratings.pivot(
    index="userId",
    columns="movieId",
    values="rating",
)
user_movie_matrix = user_movie_matrix.fillna(0)
user_movie_matrix
## 2. Matrix Factorization
### 2.1 초기 세팅
#### 2.1.1 정답 R
R = user_movie_matrix.values
n_user = R.shape[0]  # 전체 유저 수
n_item = R.shape[1]  # 전체 영화 수
#### 2.1.2 잠재 요인 행렬

# 유저와 영화별로 잠재 요인 크기가 10인 행렬을 선언합니다.
K = 10
#### 2.1.3 P와 Q 랜덤 값으로 초기화
# 유저 행렬 P와 영화 행렬 Q를 랜덤 값으로 초기화 합니다.
P = np.random.normal(size=(n_user, K))
Q = np.random.normal(size=(n_item, K))
P
Q
### 2.2 Gradient Descent를 이용한 잠재 요인 행렬 학습

# 유저 "670"이 영화 "0"에 평가한 점수를 학습하는 과정을 소개합니다.
user_id = 670
item_id = 0
#### 2.2.1 $\hat{R}$ 을 계산합니다.

# $$
# \hat{r_{AB}} = p_A^T q_B
# $$
pred = P[user_id, :].dot(Q[item_id, :].T)
pred
#### 2.2.2 $R$과 $\hat{R}$의 오차를 계산합니다.

# $$
# e_{AB} = r_{AB} - \hat{r_{AB}}
# $$
real = R[user_id, item_id]
real
error = real - pred
error
#### 2.2.3 Gradient Descent를 이용한 P와 Q를 업데이트 합니다.


# $$
# p'_A = p_A + 2\gamma e_{AB}q_{B}
# $$
# $$
# q'_B = q_B + 2\gamma e_{AB}p_{A}
# $$
learning_rate = 0.01
dp = 2 * error * Q[item_id, :]
dq = 2 * error * P[user_id, :]
P[user_id, :] += learning_rate * dp
Q[item_id, :] += learning_rate * dq
P[user_id]
# 업데이트된 P와 Q를 이용해 오차가 감소했음을 알 수 있습니다.
pred = P[user_id, :].dot(Q[item_id, :].T)
error = real - pred
error
#### 2.2.4 업데이트 과정을 반복합니다.
epochs = 10
real = R[user_id, item_id]

for epoch in range(epochs):
    pred = P[user_id, :].dot(Q[item_id, :].T)
    error = real - pred

    dp = 2 * error * Q[item_id, :]
    dq = 2 * error * P[user_id, :]

    P[user_id, :] += learning_rate * dp
    Q[item_id, :] += learning_rate * dq

    print(f"Epoch{epoch}: {round(error, 3)}")
### 2.3 전체 데이터를 이용해 P와 Q 업데이트
K = 10

P = np.random.normal(size=(n_user, K))
Q = np.random.normal(size=(n_item, K))

epochs = 5
learning_rate = 0.01

for epoch in range(1, epochs + 1):
    total_error = 0
    iteration = 0

    # 모든 유저에 대해 반복
    for user_id in range(n_user):
        # 모든 아이템에 대해 반복
        for item_id in range(n_item):

            real = R[user_id, item_id]

            # 평가하지 않은 경우 제외
            if real == 0:
                continue

            # P와 Q 업데이트
            pred = P[user_id, :].dot(Q[item_id, :].T)
            error = real - pred

            dp = 2 * error * Q[item_id, :]
            dq = 2 * error * P[user_id, :]

            P[user_id, :] += learning_rate * dp
            Q[item_id, :] += learning_rate * dq

            total_error += (error ** 2)
            iteration += 1

    print(f"Epoch {epoch}: {round(np.sqrt(total_error / iteration), 5)}")

### 2.4 영화 추천하기

# 모든 영화에 대해서 점수를 예측하고 예측 평가 점수가 높은 영화를 유저에게 추천합니다.
user_id = 124
# 유저 "124" 잠애요인에 모든 영화의 잠재요인을 곱해 평점 예측
prediction = P[[user_id], :].dot(Q.T)[0]
# 영화 아이디별 예측 평가 점수를 내림차순으로 정렬
prediction = pd.Series(
    data=prediction,
    index=user_movie_matrix.columns,
).sort_values(ascending=False)
prediction
# 아직 평가하지 않은 영화만 추출
prediction = prediction[user_movie_matrix.loc[user_id] == 0]
# 예측 평가 점수 상위 10개의 영화 아이디 추출
recommend = prediction[:10].index
recommend
movies.loc[recommend]

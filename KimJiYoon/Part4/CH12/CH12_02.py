# 샘플 데이터와 유사도 함수 실습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

## 1. 유클리디안 유사도
# 유사도를 계산할 유저를 인덱스로 아이템을 컬럼으로 하는 데이터를 정의합니다.
# 강의와 같이 유저가 평가하지 않은 아이템에 대해서는 결측값으로 표시되어 있습니다.

# 1.1 Sample Data
data = [
    [1., None, 1., None],
    [None, 1., 1., None],
    [1., None, 1., 1.],
]

df = pd.DataFrame(
    data=data,
    index=["userA", "userB", "userC"],
    columns=["itemA", "itemB", "itemC", "itemD"],
)
df

# 1.2 결측값 제거
# 결측값을 0으로 대체
df = df.fillna(0)
df

# 1.3 유클리디안 유사도 계산
# 유저-아이템 평가 행렬에서 유저별로 유클리디안 유사도를 계산
# 유클리디안 유사도 = \frac{1}{\text{유클리디안 거리} + \text{1e-5}}
# `sklearn.metrics.pairwise`의 `euclidean_distances`를 이용해 유클리디안 거리를 계산합니다.

from sklearn.metrics.pairwise import euclidean_distances

euclidean_distances(
    X=df.loc[["userA"]],
    Y=df.loc[["userB"]]
)

# `euclidean_distances`에 X와 Y를 입력할 경우, X와 Y의 각 Row끼리 유클리디안 거리를 계산합니다.
# 유저A와 유저B의 Row를 각각 X와 Y로 입력하면 두 유저의 유클리디안 거리를 계산할 수 있습니다.
euclidean_distances(df)
# `euclidean_distances`에 X만 입력할 경우, X의 모든 Row사이의 유클리디안 거리를 계산합니다.
# 전체 데이터를 입력할 경우 모든 유저 사이의 유클리디안 거리를 계산할 수 있습니다.
# 유클리디안 거리에 역수를 취해 유클리디안 유사도를  계산합니다.
distance = euclidean_distances(df)
similarity = 1 / (distance + 1e-5)# 0을 방지하기 위해 아주 작은수 1e-5더해줌
similarity

# 2. 코사인 유사도
# 2.1 코사인 유사도 계산
# 유클리디안 유사도 계산에 사용한 데이터를 이용해 코사인 유사도를 계산해 보겠습니다.
# `sklearn.metrics.pairwise`의 `cosine_similarity`를 이용해 계산할 수 있습니다.
# 유클리디안 유사도 계산과 마찬가지로 X와 Y를 각각 입력할 수 있고, X만 입력할 수 있습니다.

from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity(
    X=df.loc[["userA"]],
    Y=df.loc[["userB"]],
)
# 모든 유저 사이의 코사인 유사도를 계산하면 다음과 같습니다.
cosine_similarity(df)

# 3. 피어슨 유사도
# 아이템에 대한 유저의 선호도를 반영한 행렬에 대해 피어슨 유사도를 적용해 보겠습니다.

# 3.1 Sample Data
# 새로운 데이터를 정의합니다.
data = [
    [4., 5., 4., 3.],
    [3., 4., 3., 2.],
    [4., 4., 5., 3.],
]

df = pd.DataFrame(
    data=data,
    index=["userA", "userB", "userC"],
    columns=["itemA", "itemB", "itemC", "itemD"],
)

df
# 3.2 피어슨 유사도 계산 - 유저의 리뷰데이터로 진행
# 피어슨 유사도는 `numpy.corrcoef`를 이용해 계산할 수 있습니다.
# `numpy.corrcoef`는 데이터의 각 Row별로 유사도를 계산합니다.
np.corrcoef(df)

# 3.3 코사인 유사도 계산
# 피어슨 유사도는 유저 또는 아이템 별로 특성을 제거한 데이터에 코사인 유사도를 적용한 것과 같습니다.
# 유저별로 선호도 평균을 계산하고, 기존 데이터에서 유저별 선호도를 제거합니다.
# 1. `df.mean(axis=1)`은 각 행에 대해 평균을 계산합니다.
# 2. `df1.sub(df2, axis=0)`은 인덱스를 기준으로 두 데이터의 차를 계산합니다.
# 계산이 실제로 맞는지 확인
df.mean(axis=1)

user_mean = df.mean(axis=1)
df_sub = df.sub(user_mean, axis=0)
df_sub

cosine_similarity(df_sub)

# 4. 자카드 유사도
# 4.1 Sample Data
# 유저마다 다른 아이템에 대해 선호도를 평가한 데이터를 정의합니다.
data = [
    [4., 0., 4., 3., 0.],
    [3., 4., 0., 2., 0.],
    [0., 0., 4., 5., 3.],
]

df = pd.DataFrame(
    data=data,
    index=["userA", "userB", "userC"],
    columns=["itemA", "itemB", "itemC", "itemD", "itemE"],
)
df

# 4.2 자카드 유사도 계산
# `sklearn.metrics`의 `jaccard_score`를 이용해 자카드 유사도를 계산할 수 있습니다.
from sklearn.metrics import jaccard_score
# `jaccard_score`는 값의 크기는 무시하고 아이템의 유무를 0과 1로 표현합니다.
# 0보다 큰 값을 가지는 경우 선호도를 평가한 것을 표현하기 위해 1로 대체합니다.
df[df > 0] = 1
df

# `jaccard_score`는 비교하는 두 유저의 값을 각각 입력해야합니다.
jaccard_score(
    df.loc["userB"],
    df.loc["userC"],
)

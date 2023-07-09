import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

import squarify
from mlxtend.frequent_patterns import apriori # https://jaaamj.tistory.com/114
from mlxtend.frequent_patterns import association_rules
from wordcloud import WordCloud

# Settings Warning and Plot Hangul
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'AppleGothic'
# Part4. [실습5] 마케팅을 위한 연관 규칙 분석
# https://www.kaggle.com/datasets/hemanthkumar05/market-basket-optimization --> 데이터 코드는 여기서 받을것
# 01. 데이터 소개 및 분석프로세스 수립
# 02. 데이터 준비를 위한 EDA 및 전처리
# 0. 데이터 불러오기
data = pd.read_csv('../../../comFiles/Market_Basket_Optimisation.csv', header=None)
data.head()
data.shape
# 랜덤 샘플 데이터 보기
data.sample(10)

# 1.데이터 탐색
# 컬럼 별 데이터 타입 알아보기
data.info()
# 2. 데이터 통계값
data.describe()
# 3. 인기 판매 상품 시각화
# Word Cloud
plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color='white', width=1200, height=1200, max_words=121).generate(str(data[0]))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('인기 판매 상품들', fontsize=20)
plt.show()

# 히스토그램
plt.rcParams['figure.figsize'] = (18, 7)
color = plt.cm.copper(np.linspace(0, 1, 40))
data[0].value_counts().head(40).plot.bar(color=color)
plt.title('인기 판매 상품들', fontsize=20)
plt.xticks(rotation=90)
plt.grid()
plt.show()

# 트리맵
# 상위 50개 항목만 쓴다
y = data[0].value_counts().head(50).to_frame()
y.index
plt.rcParams['figure.figsize'] = (20, 20)
color = plt.cm.cool(np.linspace(0, 1, 50))
# 트리맵을 만들어주는 함수
squarify.plot(sizes=y.values, label=y.index, alpha=.8, color=color)
plt.title('인기 판매 상품들 트리맵')
plt.axis('off')
plt.show()

# 4) 결측값
# 5) 중복값
# 2. 데이터 전처리
# 1) 데이터 컬럼명 수정
# 동일한 크기의 리스트에 각 손님들의 쇼핑 목록을 넣기
trans = []
for i in range(0, 7501):
    trans.append([str(data.values[i, j]) for j in range(0, 20)])

# numpy array 변환
trans = np.array(trans)
print(trans.shape)

# 2) Transaction Encoder 적용
# 해당 인코더 URL
# https://rasbt.github.io/mlxtend/user_guide/preprocessing/TransactionEncoder/
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
data = te.fit_transform(trans)
data = pd.DataFrame(data, columns=te.columns_)
data.shape
data.head()
# > 기존 컬럼 대상으로 필터링
len(y.index)
data = data[y.index]

# 03. 연관 규칙 분석
# * 지지도 (support) : 항목에 대한 거래수 / 전체 거래수
# * 신뢰도 (confidence) : 조건과 결과 항목을 동시에 포함하는 거래수 / 조건 항목을 포함한 거래수
# * 향상도 (lift) : lift(C,A) = support(C->A) / (support[A] * support[C]) = confidence(C->A) / support(A)
# * 1보다 크면 연관성 다수 1보다 작으면 연관성 줄어듬
# * 예) 거래1 : (A,B,C) / 거래2 : (A,C) / 거래3 : (A,D) / 거래4 : (E,F,G) 일 때,
# > lift(C,A) = (2/2)/(3/4) = 1.3333
# 연관 규칙 관련 블로그 : https://zephyrus1111.tistory.com/119

# 1) Apriori 알고리즘 적용 절차
# 1. 모든 거래에서 발생하는 모든 항목에 대한 빈도 테이블을 생성
data.head()
# 2. support 가 임의의 값보다 큰 것들로 필터링
# 3. 중요 항목의 모든 가능한 조합을 만들기
# min_support : 최소 지지도는 0.01로 셋팅
frequent_itemsets = apriori(data, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False)
# 4. 모든 조합의 발생 횟수 계산
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
# 필터링 처리
# 발생 회수가 3이고 지지도가 0.01 이상인 경우
frequent_itemsets[(frequent_itemsets['length'] == 3) & (frequent_itemsets['support'] >= 0.01)]
# 발생 회수가 2이고 지지도가 0.01 이상인 경우
frequent_itemsets[(frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.01)]
# 발생 회수가 1이고 지지도가 0.1 이상인 경우
frequent_itemsets[(frequent_itemsets['length'] == 1) & (frequent_itemsets['support'] >= 0.1)]

# 2) Association Rules 적용
df_ar = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.1)
df_ar.sort_values("confidence", ascending=False)

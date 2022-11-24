#CH05_02. 데이터 전처리

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')

df.head() #상위 5개 데이터 확인
df.describe() #데이터 요약
df.info() #데이터 정보

#문제: 처음부터 800번까지의 데이터를 학습 데이터로 이용하고, 
#나머지 데이터를 테스트 데이터로 이용하여 모델간의 결과를 비교하여라.

train_df = df[:800] #처음(1번)부터 800번까지 train_df
test_df = df[800:] #801번부터 끝까지 test_df

print(len(train_df)) #len(이름): length
print(len(test_df))

#pclass 와 survived 의 관계(관계 있음)
train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived', ascending=False)
'''
pclass와 survived를 pclass를 기준으로 그룹핑하고, index는 false로(?) 평균 생존률로 내림차순
1에 가까울 수록 생존률이 높다고 판단
'''

#sex 와 survived 의 관계(관계 있음)
train_df[["sex", "survived"]].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False)

#parch 와 survived 의 관계(관계가 적음)
train_df[["parch", "survived"]].groupby(['parch'], as_index=False).mean().sort_values(by='survived', ascending=False)

#sibsp 와 survived 의 관계(관계가 적음)
train_df[["sibsp", "survived"]].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived', ascending=False)

#age 와 survived 의 관계
sns.histplot(data = train_df, x = 'age', bins = 20, hue = 'survived') #히스토그램
plt.show()

a = sns.FacetGrid(train_df, col='survived') #FacetGrid: subplot처럼 그림 그리는 구간을 나눈 것
a.map(plt.hist, 'age', bins=20)
plt.show()

#pclass 에 따른 age 별 survived 유무
a = sns.FacetGrid(train_df, col='survived', row='pclass') #row 추가
a.map(plt.hist, 'age', bins=20)
plt.show()

#필요 없는 필드 삭제
names = train_df.columns
print(names)

train_df = train_df.drop(names[4:], axis = 1) #drop 지우기, 4개까지 남기고 그 이후 다 지움
train_df.head()

test_df = test_df.drop(names[4:], axis = 1)
test_df.head()

#결측값 확인
print(train_df.isnull().sum())
print(test_df.isnull().sum())

#age 평균으로 age 결측값 채우기
'''
*만약 pclass 별 age 의 평균으로 채우고 싶다면 아래 주석 처리된 코드 사용
train_df["age"] = train_df.groupby(['pclass']).age.transform(lambda x: x.fillna(x.mean()))
test_df["age"] = test_df.groupby(['pclass']).age.transform(lambda x: x.fillna(x.mean()))
'''

train_df.fillna(train_df.mean()[['age']], inplace = True)
test_df.fillna(test_df.mean()[['age']], inplace = True)

print(train_df.isnull().sum())
print(test_df.isnull().sum())

#성별 인코딩
map_dict = {'female' : 0, 'male' : 1} 
'''
학습을 하려면 데이터가 수치화되어야 함
map_dict라는 딕셔너리로 만들어 0, 1로 변환
'''

train_df['sex'] = train_df['sex'].map(map_dict).astype(int)
test_df['sex'] = test_df['sex'].map(map_dict).astype(int)

train_df.head()

#나이 분류

def function1(x):
    if x < 20:
        return 1
    elif x < 40:
        return 2
    elif x < 60:
        return 3
    else:
        return 4

train_df['age'] = train_df['age'].apply(function1)
test_df['age'] = test_df['age'].apply(function1)

train_df.head()
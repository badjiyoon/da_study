# CH03_08. 결측값 처리

import pandas as pd
import numpy as np

df=pd.DataFrame({'a': [1, 1, 3, 4 ,5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
#왜 b열의 값은 nan을 제외하고 모두 2.0, 3.0 형식으로 나올까?

#결측 유무 확인
df.isnull() #빈값=nan 이 있는지 확인, T/F로 리턴

#결측값 개수 확인
df.isnull().sum() #각 열에 nan 개수 출력
df.isnull().count()

'''
count를 하게 되면 T/F로 출력된 데이터프레임의 
빈값을 제외하고 모두 더하는 것으로 인식 -> nan이 아니면 세는 거니까 모두 5로 출력
sum을 하게 되면 T=1, F=0으로 판단하므로, T의 개수를 출력하게 됨
'''

#결측값이 포함된 열/행 지우기
df.dropna() #결측값이 포함된 행 지우기
df.dropna(inplace=True)

df=pd.DataFrame({'a': [1, 1, 3, 4 ,5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})
df.dropna(axis=1) #dropna는 행 지우기인데, axis=1을 하면 열 지우기로 바뀜
df.dropna(axis=0) #axis에 0이면 없는 거니까 행 지우기가 됨
df.dropna(axis=1, inplace=True)

#결측값을 다른 값으로 대체하기
df=pd.DataFrame({'a': [1, 1, 3, 4 ,5], 'b': [2, 3, np.nan, 3, 4], 'c': [3, 4, 7, 6, 4]})

#특정값으로 대체하기
df.fillna(0) #0으로 대체
df.fillna(0, inplace=True)

#앞이나 뒤의 숫자로 바꾸기
df=pd.DataFrame({'a': [1, 1, 3, 4 ,np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})

#1) 뒤의 값으로 채우기 (bfill=backward fill)
df.fillna(method='bfill') #a열 5행은 뒤의 값이 없으므로 여전히 nan으로 출력

#2) 앞의 값으로 채우기 (ffill=forward fill)
df.fillna(method='ffill') #c열 1행은 앞의 값이 없으므로 여전히 nan으로 출력

#limit 설정
df.fillna(method='ffill', limit=1) #앞의 값을 한 번만 가져옴 (밀려서 앞의 값 두 번 가져오기 안됨)

#문제: 데이터프레임에 존재하는 결측값들을 뒤의 값으로 대체한 이후 앞의 값으로 대체하시오.
df=pd.DataFrame({'a': [1, 1, 3, 4 ,np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 7, 6, 4]})
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

#평균으로 대체
df=pd.DataFrame({'a': [1, 1, 3, 4 ,np.nan], 'b': [2, 3, np.nan, np.nan, 4], 'c': [np.nan, 4, 1, 1, 4]})
df.mean()['a'] #a열 나머지 데이터의 평균
df.fillna(df.mean()['a']) #a열 나머지 데이터의 평균으로 nan 채우기
df.fillna(df.mean()) #각 열의 nan 외에 데이터들의 평균으로 채우기
df
#문제: b, c의 결측값들을 데이터프레임의 전체 값의 평균으로 치환하시오.
df.fillna(df.mean()[['b', 'c']])

type(df['a'])
type(df[['b', 'c']])
df.mean()['a'] #a열의 시리즈 데이터의 평균 -> 값이 1개 출력
df.mean()[['b', 'c']] #b, c열의 각각 데이터의 평균 -> 값이 2개 출력

'''
['a'] 는 시리즈 -> 시리즈의 평균은 결과 1개 출력
[['b', 'c']] 는 데이터프레임 -> 데이터프레임의 평균은 각 열의 평균으로 열 개수만큼 출력
'''

df.fillna(df.mean()['a']) 
df.fillna(df.mean()[['b', 'c']])

'''
fillna(데이터 1개) -> 하나의 데이터로 전체 nan을 채움
fillna(데이터 2개) -> 각각의 데이터를 해당하는 열의 nan만 채움
'''
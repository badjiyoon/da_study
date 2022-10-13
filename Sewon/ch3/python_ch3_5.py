# CH03_05. loc와 iloc을 이용한 원하는 위치의 데이터 추출

import pandas as pd
df=pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})
df

#문제: a, b열을 추출하시오.
 
'''
시리즈는 한 번에 한 줄만 뽑을 수 있음. 
두 줄을 뽑으려면 데이터프레임 뒤에 이중 대괄호
'''

df[['a', 'b']] #이중 대괄호니까 각 열의 데이터들을 나열한 데이터프레임
df['a']
type(df)
type(df[['a', 'b']]) 

#문제: 첫 번째 행의 데이터를 출력하시오.
df[0]
df.loc[0] #0=첫 번째 행 (not 칼럼명)
df.loc[2:4] #슬라이싱도 가능

#인덱스가 문자로 이루어진 데이터 프레임 생성
index=['a', 'b', 'd', 'c', 'e', 'f', 'g', 'g', 'h', 'i']
df=pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]}, index=index)
df

df.loc[0] #인덱스명이 달라졌기 때문에 오류
df.loc['g']
df.loc['c'] #행의 이름에 대응되는 것으로 추출
df.loc['c':] #슬라이싱 가능

#열이 a, c이며 인덱스가 g, i인 데이터를 출력하시오.
df.loc[['g', 'i'], ['a', 'c']] # 행, 열 순서

#문제: 처음부터 5번째까지의 데이터와 첫 번째 열과 세 번째 열의 데이터를 추출하시오.
df=pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]})
df.iloc[:5, [0, 2]] #행렬 나타내듯이

#문제: 처음부터 5번째까지의 데이터와 첫 번째 열과 세 번째 열의 데이터를 추출하시오.
index=['a', 'b', 'd', 'c', 'e', 'f', 'g', 'g', 'h', 'i']
df=pd.DataFrame({'a': [i for i in range(1, 11)], 'b': [i for i in range(11, 21)], 'c': [i for i in range(21, 31)]}, index=index)
df
df.iloc[:5, [0,2]]

'''
*loc와 iloc의 차이
loc: 라벨값으로 인덱싱
iloc: 정수값으로 인덱싱

*정확한 칼럼명, 로우명으로 값을 추출하고 싶을 때는 loc
칼럼명, 로우명은 모르지만 위치값으로 추출하고 싶을 때는 iloc

*둘 다 행, 열 순서로 작성
'''
# CH03_07. 정렬

import pandas as pd

#인덱스 기준 정렬
df=pd.DataFrame({'a': [2, 3, 2, 7, 4], 'b': [2, 1, 3, 5, 3], 'c': [1, 1, 2, 3, 5]})

df.sort_index() #내림차순일 때는 ascending=False
df.sort_index(ascending=False) #결과를 저장하고 싶으면 inplace=True
df.sort_index(ascending=False, inplace=True)

df.reset_index() #인덱스 초기화는 reset_index

'''
바꾸기 전에 기존 인덱스도 같이 보여줌
reset_index에서 drop하면 기존 인덱스도 안보여줌
'''

df.reset_index(drop=True) #데이터프레임이 변경된 것은 아님
df.reset_index(drop=True, inplace=True)

#값 기준 정렬
df=pd.DataFrame({'a': [2, 3, 2, 7, 4], 'b': [2, 1, 3, 5, 3], 'c': [1, 1, 2, 3, 5]})

#문제: a열 기준으로 오름차순 정렬하시오.
df.sort_values('a') #a열의 값을 기준으로 오름차순 정렬
df.sort_values('a', inplace=True) 

#문제: a열 기준으로 내림차순 정렬하시오.
df.sort_values(by=['a'], ascending=False) #by가 있으나 없으나 결과가 같음
df.sort_values('a', ascending=False, inplace=True)

#문제: a, b열 기준으로 오름차순 정렬하시오.
df.sort_values(['a', 'b'])
df.sort_values(by=['a', 'b']) #by가 있으나 없으나 결과가 같음

#문제: a열 기준으로 오름차순 정렬한 이후, b열 기준으로 내림차순 정렬하시오.
df.sort_values(['a', 'b'], ascending=[True, False])
df.sort_values(by=['a', 'b'], ascending=[True, False]) #by가 있으나 없으나 결과가 같음
df.sort_values(by=['a', 'b'], ascending=[True, False], inplace=True)
df.reset_index(drop=True, inplace=True)
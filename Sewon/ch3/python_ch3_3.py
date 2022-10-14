# CH03_03. copy를 이용한 데이터 복사

import pandas as pd
df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df.columns = ['d', 'e', 'f']

#문제: 필드명 a를 '에이'로 변경하시오.
df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df.rename(columns={'a':'에이'}, inplace=True)

'''
변수명을 복제하여 반복하여 
DataFrame을 다시 만드는 수고를 덜을 수 있음
'''

df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df2 = df #변수명 복제
df
df2

df.columns=['d', 'e', 'f']
df
df2 #shallow copy

'''
shallow copy
하나의 객체에 변수명이 2개가 생기는 꼴
'''

#deep copy
import copy
df=pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6], 'c': [7, 8, 9]})
df2=copy.deepcopy(df) 
df
df2

df.columns = ['d', 'e', 'f']
df
df2

'''
deep copy
두 개의 객체에 각각 변수명이 1개씩 생기는 꼴
*데이터프레임을 변경하다가 원데이터가 오염되는 것을 방지
'''
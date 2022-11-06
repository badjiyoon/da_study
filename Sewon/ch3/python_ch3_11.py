# CH03_11. apply.map을 활용한 데이터 변환

import pandas as pd
df=pd.DataFrame({'a': [1, 2, 3, 4, 5]})

#문제: a가 2보다 작으면 '2 미만', 4보다 작으면 '4 미만', 4보다 크면 '4 이상'이 저장된 b 칼럼을 추가하시오.
df
df['b']=0
a=df[df['a']<2]
a
df['b'][a.index]='2 미만'
df
a=df[(df['a']>=2)&(df['a']<4)]
df['b'][a.index]='4 미만' #SettingWithCopyWarning: shallow copy
df
pd.set_option('mode.chained_assignment', None) #SettingWithCopyWarning 오류 끄는 코드
a=df[df['a']>=4]
df['b'][a.index]='4 이상'

#함수+apply를 이용한 해결
df
def case_function(x):
    if x<2:
        return '2 미만'
    elif x<4:
        return '4 미만'
    else:
        return '4 이상'
df['c']=df['a'].apply(case_function)
#c열에 a열에 case_function을 적용한 값을 출력
df

#문제: a가 1이면 'one', 2이면 'two', 3이면 'three', 4이면 'four', 5이면 'five'를 출력하는 칼럼 d를 만드시오.
#사용자 정의 함수를 이용한 해결 방법
def function(x):
    if x==1:
        return 'one'
    elif x==2:
        return 'two'
    elif x==3:
        return 'three'
    elif x==4:
        return 'four'
    elif x==4:
        return 'five'
df['d']=df['a'].apply(function)
df

#map을 이용한 해결 방법
a={1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five'}
df['e']=df['a'].map(a)
df
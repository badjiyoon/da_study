# CH03_09. 타입 변환

import pandas as pd
df=pd.DataFrame({'판매일': ['5/11/21', '5/12/21', '5/13/21', '5/14/21', '5/15/21'],
                 '판매량': ['10', '15', '20', '25', '30'],
                 '방문자수': ['10', '-', '17', '23', '25'],
                 '기온': ['24.1', '24.3', '24.8', '25', '25.4'],
                 '기온': ['24.1', '24.3', '24.8', '25', '25.4']})
#왜 데이터프레임의 열 부분이 set으로 되어 있을까?
#set의 특징을 가질까 싶어 같은 원소를 넣었는데 역시나 같은 원소는 한 개만 출력
#데이터프레임도 사실은 열들의 집합 개념인듯

df=pd.DataFrame({'판매일': ['5/11/21', '5/12/21', '5/13/21', '5/14/21', '5/15/21'],
                 '판매량': ['10', '15', '20', '25', '30'],
                 '방문자수': ['10', '-', '17', '23', '25'],
                 '기온': ['24.1', '24.3', '24.8', '25', '25.4']})

#타입 확인 
df.dtypes

df['판매량 보정']=df['판매량']+1 
#판매량 열에 있는 값들은 string, 연산 불가

#문제: 판매량을 정수 형태로 변환하시오.
df.astype({'판매량':'int'}) #출력할 때만 int
df.dtypes
df=df.astype({'판매량':'int'}) #df를 변경
df.dtypes

#문제: 방문자수를 숫자 타입으로 변형하시오.
df.astype({'방문자수':'int'}) #'-' 숫자가 아님
pd.to_numeric(df['방문자수']) #to numeric=숫자로
pd.to_numeric(df['방문자수'], errors='coerce') #coerce=강제하다
#에러에 대해서는 강제로 numeric으로, 숫자가 아닌 것은 NaN
df['방문자수']=pd.to_numeric(df['방문자수'], errors='coerce')
df.dtypes 

'''
*1비트당 2개의 숫자를 표현
int8 => 2^8개의 정수 표현 (-128~127)
int16 => 2^16개의 정수 표현 (-32,768~32,767)
int32 => 2^32개의 정수 표현 (-2,147,483,648~2,147,483,647)
int64 => 2^64개의 정수 표현 (-9,223,372,036,854,775,808~9,223,372,036,854,775,807)

uint8 => 2^8개의 부호 없는 정수 표현 (0~255) 
 *그레이스케일 또는 3채널 컬러 이미지를 담을 때 많이 사용
uint16 => 2^16개의 부호 없는 정수 표현 가능 (0~65,535) 
uint32 => 2^32개의 부호 없는 정수 표현 가능 (0~4,294,967,295)
uint64 => 2^64개의 부호 없는 정수 표현 가능 (0~18,446,744,073,709,551,615)

float16 => 1비트 부호, 5비트 정수, 10비트 소수
float32 => 1비트 부호, 8비트 정수, 23비트 소수
float64 => 1비트 부호, 11비트 정수, 52비트 소수
'''

df.fillna(0, inplace=True)
df
df=df.astype({'방문자수':'int'})
df.dtypes

#문제: 판매일을 datetime의 형태로 바꾸시오.
df['판매일']=pd.to_datetime(df['판매일'], format='%m/%d/%y')
df
df.dtypes
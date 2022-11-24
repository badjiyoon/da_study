# CH04_04. load_dataset 을 이용한 데이터 셋 불러오기
#load_dataset 사용법. : sns.load_dataset('데이터 셋 이름')

import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd

'''
seaborn: matplot 라이브러리로 만들어진 효율적인 패키지
그래프를 matplot보다 예쁘게 그릴 수 있다는 등의 장점이 있음
'''

#예제1 : iris
df = sns.load_dataset('iris')
df.head()

#문제 : iris 데이터 셋의 petal_length 와 petal_width 를 이용하여 산점도를 그리시오. 
# (scatter: 흩뿌리다, 산점도)
plt.scatter(df['petal_length'], df['petal_width']) #matplotlib
plt.show()

sns.scatterplot(data = df, x = 'petal_length', y = 'petal_width') #seaborn
plt.show()

#matplotlib 과 호환: matplotlib으로 만들어졌기 때문에 호환됨
sns.scatterplot(data = df, x = 'petal_length', y = 'petal_width')
plt.title('iris')
plt.show()
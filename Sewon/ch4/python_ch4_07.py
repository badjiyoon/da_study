# CH04_07. scatterplot을 이용한 산점도 그리기

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(10, 5)})
import pandas as pd

#문제 : iris 데이터 셋을 이용하여 sepal_length, sepal_width 별 산점도를 작성하시오.

df = sns.load_dataset('iris')
df.head()
df.isnull().sum()
sns.scatterplot(data = df, x = 'sepal_width', y = 'sepal_length')
plt.show()

#hue를 이용한 종 분류
sns.scatterplot(data = df, x = 'sepal_width', y = 'sepal_length', hue = 'species')  #꽃 종류별로 나타냄
plt.show()

#추가 문제 : petal_length 의 값을 3 미만, 5 미만, 5 이상으로 분류하여 표시하여라.
df['petal_length'].unique() #unique가 뭐였더라? 
def function1(x):
    if x < 3:
        return 's'
    elif x < 5:
        return 'm'
    else:
        return 'l'
df['petal_length2'] = df['petal_length'].apply(function1)
df.head()
sns.scatterplot(data = df, x = 'sepal_width', y = 'sepal_length', hue = 'species', style = 'petal_length2')
#style: 점의 모양 옵션
plt.show()

#점의 크기로 분류
def function2(x):
    if x < 3:
        return 3
    elif x < 5:
        return 5
    else:
        return 7
df['petal_length3'] = df['petal_length'].apply(function2)
df.head()
sns.scatterplot(data = df, x = 'sepal_width', y = 'sepal_length', hue = 'species', size = 'petal_length3') 
#size: 점의 크기 옵션
plt.show()

#relplot 을 이용하면 카테고리 별로 따로 그릴 수 있음
sns.relplot(data = df, x = 'sepal_width', y = 'sepal_length', col = 'species')
#relplot: 모든 변수에 따라 그래프 따로 그림
plt.show()

#lmplot 을 사용하면 회귀선을 그릴 수 있음
sns.lmplot(data = df, x = 'sepal_width', y = 'sepal_length', hue = 'species')
plt.show()
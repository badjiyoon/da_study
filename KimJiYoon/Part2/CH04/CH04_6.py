import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd
sns.set(rc={'figure.figsize': (10, 5)})

df = sns.load_dataset('titanic')
# 성별 인원 수 처리
print(df.head())
print(df.describe())
print(df.isnull().sum())
print(df.groupby(by='sex')['sex'].count())
# sns.countplot(data=df, x='sex')
# sns.countplot(data=df, y='sex')
# 성별 인원수를 객실 등급별로 시각화
print(df[['sex','class']])
print(df.groupby(by=['sex', 'class'])['sex'].count())
# sns.countplot(data=df, x='sex', hue='class')
# palette를 사용한 색상조정
sns.countplot(data=df, x='sex', hue='class', palette='flare')
plt.show()

import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (10, 5)})

df = sns.load_dataset('flights')
print(df.head())
# 결측치 확인
print(df.isnull().sum())
# sns.lineplot(data=df, x='year', y='passengers')
# xtick에 전체 년도 표시
# sns.lineplot(data=df, x='year', y='passengers')
# plt.xticks(df['year'])
# xtick에 기울기 표시
# sns.lineplot(data=df, x='year', y='passengers')
# plt.xticks(df['year'], rotation=45)
# 월별로 그리기
sns.lineplot(data=df, x='year', y='passengers', hue='month', palette='Set2')
# 선표현 방법 변경
sns.lineplot(data=df, x='year', y='passengers', hue='month', palette='Set2', style='month')
plt.show()

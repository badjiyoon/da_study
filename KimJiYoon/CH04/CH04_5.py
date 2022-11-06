import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
sns.set(rc={'figure.figsize': (10, 5)})
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

movie_title = ['크루엘라', '극장판 귀멸의 칼날: 무한열차편', '학교 가는 길']
audience = [664308, 2099131, 20067]

data = {'영화제목': movie_title, '누적관객': audience}
df = pd.DataFrame(data)
print(df)
"""
sns.barplot(data=df, x='영화제목', y='누적관객')
# 누적관객수 별로 그리기
sns.barplot(data=df, x='영화제목', y='누적관객',
            order=df.sort_values('누적관객').영화제목)

import matplotlib.ticker as mticker

chart = sns.barplot(data=df, x='영화제목', y='누적관객',
            order=df.sort_values('누적관객', ascending=False).영화제목)

# 워닝 처리
ticks_labels = chart.get_yticks().tolist()
chart.yaxis.set_major_locator(mticker.FixedLocator(ticks_labels))
ylabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_yticks() / 10000]
chart.set_yticklabels(ylabels)
"""
import matplotlib.ticker as mticker

chart = sns.barplot(data=df, x='누적관객', y='영화제목',
            order=df.sort_values('누적관객', ascending=False).영화제목, color='blue')

# 워닝 처리
ticks_labels = chart.get_xticks().tolist()
chart.xaxis.set_major_locator(mticker.FixedLocator(ticks_labels))
xlabels = ['{:,.0f}'.format(i) + '만 명' for i in chart.get_xticks() / 10000]
chart.set_xticklabels(xlabels)

plt.xlabel('누적관객', fontsize=15)
plt.ylabel('영화제목', fontsize=15)
plt.title('영화 별 누적관객수', fontsize=15)
plt.show()

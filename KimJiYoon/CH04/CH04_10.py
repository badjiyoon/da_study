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
# plt.pie(df['누적관객'])
# plt.pie(df['누적관객'], labels=df['영화제목'])
# plt.pie(df['누적관객'], labels=df['영화제목'], autopct='%0.2f%%')
# colors_list = ['red', 'green', 'blue']
# colors_list = ['#08080', '#0CDCDC', '#FFF80C']
# plt.pie(df['누적관객'], labels=df['영화제목'], autopct='%0.2f%%', colors=colors_list)
explode_list = [0, 0, 1, 0]
plt.pie(df['누적관객'], labels=df['영화제목'], autopct='%0.2f%%', explode=explode_list)
plt.show()


import matplotlib.pyplot as plt
plt.rcParams['font.family']=['NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd

movie_title = ['크루엘라', '극장판 귀멸의 칼날: 무한열차편', '학교 길']
audience = [664308, 2099131, 20067]

data = {'영화제목' : movie_title, '누적관객' : audience}
df = pd.DataFrame(data)
plt.pie(df['누적관객'])
plt.pie(df['누적관객'], labels = df['영화제목']); 
plt.show()
plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%'); 
colors_list = ['#F08080', '#DCDCDC', '#FFF8DC']

plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%', colors = colors_list);
explode_list = [0, 0.1, 0] #

plt.pie(df['누적관객'], labels = df['영화제목'], autopct = '%0.2f%%', explode = explode_list);
plt.show()
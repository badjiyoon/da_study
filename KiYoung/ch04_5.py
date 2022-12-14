
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

movie_title = ['크루엘라', '극장판 귀멸의 칼날', '학교가는길'] 
audience = [664308, 222222, 335522]

data = {'영화제목' : movie_title, '누적관객' : audience}
df = pd.DataFrame(data)
df

sns.barplot(data = df, x = '영화제목', y = '누적관객')
sns.barplot(data = df, x = '영화제목', y = '누적관객', order = df.sort_values('누적관객', ascending = Fasle).영화제목)
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import pandas as pd

sns.set(rc={'figure.figsize': (12, 2)})

df = sns.load_dataset('flights')
print(df.head())
# 결측치 확인
print(df.isnull().sum())
 
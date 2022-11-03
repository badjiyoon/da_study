# -*- coding: utf-8 -*-
"""CH04_4.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bwDKqAMtYU3RpPDvratVz0iijFpPf3QG
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = sns.load_dataset('iris')
df.head()

plt.scatter(df['petal_length'], df['petal_width'])
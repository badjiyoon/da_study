import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns

df = sns.load_dataset('iris')
print(df.head())
profile = ProfileReport(df)
profile.to_notebook_iframe()

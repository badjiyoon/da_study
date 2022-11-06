import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
df = sns.load_dataset('iris')
profile = ProfileReport(df)

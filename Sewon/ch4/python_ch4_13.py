# CH04_13. pandas-profiling 을 통한 EDA

'''
아래 셀 실행 이후 런타임 -> 런타임 다시 시작
!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
Installing collected packages: jinja2, colorama, attrs, tqdm, matplotlib, 
imagehash, visions, statsmodels, phik, missingno, pandas-profiling

!pip install -U pandas-profiling
Installing collected packages: imagehash, visions, statsmodels, phik, missingno, pandas-profiling

!pip install ipython
!pip install ipywidgets
Installing collected packages: pywin32, widgetsnbextension, tornado, pyzmq, psutil, nest-asyncio, 
jupyterlab-widgets, jupyter-core, entrypoints, debugpy, jupyter-client, ipykernel, ipywidgets
'''

import numpy as np
import pandas as pd
from pandas_profiling import ProfileReport
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib as mpl

df = sns.load_dataset('iris')

profile = ProfileReport(df)
profile.to_notebook_iframe()

'''
ImportError: cannot import name '_check_savefig_extra_args' from 'matplotlib.backend_bases' 
(C:\Users\sewon\AppData\Local\Programs\Python\Python310\lib\site-packages\matplotlib\backend_bases.py)
*해당 파일에 있는데 왜 안된다는건지 모르겠음
'''
# LIME 실습

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)
## 1. Data
### 1.1 Data Load
from sklearn.datasets import fetch_20newsgroups


categories = ["alt.atheism", "soc.religion.christian"]
newsgroups_train = fetch_20newsgroups(subset="train", categories=categories)
newsgroups_test = fetch_20newsgroups(subset="test", categories=categories)
class_names = ["atheism", "christian"]
### 1.2 Data Preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(lowercase=False)
train_vectors = vectorizer.fit_transform(newsgroups_train.data)
test_vectors = vectorizer.transform(newsgroups_test.data)
## 2. Model
### 2.1 모델 학습
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=500)
rf.fit(train_vectors, newsgroups_train.target)
### 2.1 예측
text_instance, instance_label = newsgroups_test.data[6], newsgroups_test.target[6]
print(text_instance)
instance_label
rf.predict(test_vectors[0])
## 3.LIME
### 3.1 Pipeline
from sklearn.pipeline import make_pipeline

c = make_pipeline(vectorizer, rf)
c.predict([text_instance])
c.predict_proba([text_instance])
### 3.2 Explainer
from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=class_names)
### 3.3 Explain
explain = explainer.explain_instance(text_instance, c.predict_proba)
list
explain.as_list()
html
explain.show_in_notebook(text=False)
pyplot
explain.as_pyplot_figure()
plt.show()
notebook
explain.show_in_notebook(text=True)

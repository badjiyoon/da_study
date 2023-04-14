# 샘플 데이터와 Stacking Classification

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
# 1.1 Sample Data
from sklearn.datasets import make_classification

data, label = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2021
)

# 1.2 Data EDA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
plt.scatter(pca_data[:,0], pca_data[:,1], c=label)
plt.show()

# 1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)
# 2. 개별 모델의 성능
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'lr': LogisticRegression(),
    'knn': KNeighborsClassifier(),
    'tree': DecisionTreeClassifier(),
    'svm': SVC(),
    'bayes': GaussianNB(),
}

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
names = []
results = []
for name, model in models.items():
    result = cross_val_score(model, train_data, train_label, cv=3, scoring="accuracy")
    names += [name]
    results += [result]

results
plt.figure(figsize=(8, 8))
plt.boxplot(results, labels=names)
plt.show()

for name, model in models.items():
    model.fit(train_data, train_label)
    test_pred = model.predict(test_data)
    acc = accuracy_score(test_label, test_pred)
    print(f"Model {name} test accuracy is {acc:.4}")

# 3. Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
models.keys()
stacking = StackingClassifier(
    estimators=list(models.items()),
    final_estimator=LogisticRegression(),
    cv=3
)
stacking_result = cross_val_score(stacking, train_data, train_label, cv=3, scoring="accuracy")
stacking_result
all_result = []
all_result.extend(results)
all_result.append(stacking_result)

plt.figure(figsize=(8, 8))
plt.boxplot(all_result, labels=names + ["stacking"])
plt.show()

for name, model in models.items():
    test_pred = model.predict(test_data)
    acc = accuracy_score(test_label, test_pred)
    print(f"Model {name} test accuracy is {acc:.4}")

stacking.fit(train_data, train_label)
stacking_pred = stacking.predict(test_data)
stacking_acc = accuracy_score(test_label, stacking_pred)
print(f"Model stacking test accuracy is {stacking_acc:.4}")

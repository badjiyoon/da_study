# 샘플 데이터와 Stacking Classification
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


## 3. Stacking
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
np.random.seed(2021)
## 1. Data
### 1.1 Sample Data
from sklearn.datasets import make_classification

data, label = make_classification(
    n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=2021
)
data.shape, label.shape

### 1.2 Data EDA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca_data = pca.fit_transform(data)
plt.scatter(pca_data[:, 0], pca_data[:, 1], c=label)

### 1.3 Data Split
from sklearn.model_selection import train_test_split

train_data, test_data, train_label, test_label = train_test_split(
    data, label, train_size=0.7, random_state=2021
)

## 2. 개별 모델의 성능
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

models = {
    'lr': LogisticRegression(),
    'knn': DecisionTreeClassifier(),
    'svm': KNeighborsClassifier(),
    'tree': SVC(),
    'bayes': GaussianNB(),
}
stacking = StackingClassifier(
    estimators=list(models.items()),
    final_estimator=LogisticRegression(),
    cv=3
)

stacking_result = cross_val_score(stacking, train_data, train_label, cv=3, scoring="accuracy")
stacking_result

stacking.fit(train_data, train_label)
stacking_pred = stacking.predict(test_data)
stacking_acc = accuracy_score(test_label, stacking_pred)

print(f"Model stacking test accuracy is {stacking_acc:.4}")
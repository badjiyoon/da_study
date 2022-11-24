#CH05_03. 머신러닝 모델 구성 및 결과 검증

from sklearn.tree import DecisionTreeClassifier
import pandas as pd

X_train = train_df.drop(["survived"], axis=1)
Y_train = train_df["survived"]
X_test  = test_df.drop("survived", axis=1)
Y_test = test_df["survived"]

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
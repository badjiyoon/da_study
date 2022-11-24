import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = sns.load_dataset('titanic')
print(df.head())
print(df.describe())
print(df.info())
train_df = df[:800]
test_df = df[800:]
print(len(train_df))
print(len(test_df))
print(train_df[['pclass', 'survived']].groupby(['pclass'], as_index=False).mean().sort_values(by='survived',
                                                                                              ascending=False))
print(train_df[['sex', 'survived']].groupby(['sex'], as_index=False).mean().sort_values(by='survived', ascending=False))
print(train_df[['parch', 'survived']].groupby(['parch'], as_index=False).mean().sort_values(by='survived',
                                                                                            ascending=False))
print(train_df[['sibsp', 'survived']].groupby(['sibsp'], as_index=False).mean().sort_values(by='survived',
                                                                                            ascending=False))

sns.histplot(data=train_df, x='age', bins=20, hue='survived')
a = sns.FacetGrid(train_df, col='survived')
a.map(plt.hist, 'age', bins=20)

a = sns.FacetGrid(train_df, col='survived', row='pclass')
a.map(plt.hist, 'age', bins=20)

plt.show()

names = train_df.columns
print(names)

train_df = train_df.drop(names[4:], axis=1)
print(train_df.head())

test_df = test_df.drop(names[4:], axis=1)
print(test_df.head())

print(train_df.isnull().sum())
print(test_df.isnull().sum())

train_df.fillna(train_df.mean()[['age']], inplace=True)
test_df.fillna(test_df.mean()[['age']], inplace=True)

print(train_df.isnull().sum())
print(test_df.isnull().sum())

# 성별 인코딩
map_dict = {'female': 0, 'male': 1}
train_df['sex'] = train_df['sex'].map(map_dict).astype(int)
test_df['sex'] = test_df['sex'].map(map_dict).astype(int)


# 나이 분류
def function1(x):
    if x < 20:
        return 1
    elif x < 40:
        return 2
    elif x < 60:
        return 3
    else:
        return 4


train_df['age'] = train_df['age'].apply(function1)
test_df['age'] = test_df['age'].apply(function1)

print(train_df.head())
print(test_df.head())

from sklearn.tree import DecisionTreeClassifier

X_train = train_df.drop(['survived'], axis=1)
Y_train = train_df['survived']
X_test = test_df.drop(['survived'], axis=1)
Y_test = test_df['survived']

print(X_train.head())
print(Y_train.head())

# 모델 생성 및 학습
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

# 모델 정확도 검증
print(decision_tree.score(X_train, Y_train))
print(decision_tree.score(X_test, Y_test))

# 실제 값 예측값 비교 구현
Y_pred = decision_tree.predict(X_test)
print(Y_pred)
print(Y_test)
print(len(Y_pred))
print(len(Y_test))

Y_test_list = list(Y_test)
print(Y_pred[0])
print(Y_test_list[0])

total = 0
for i in range(len(Y_pred)):
    if Y_pred[i] == Y_test_list[i]:
        total += 1
    else:
        pass
print(total)
print(total / len(Y_pred))

from sklearn.tree import export_graphviz

export_graphviz(
    decision_tree,
    out_file='titanic.dot',
    feature_names=['pclass', 'sex', 'age'],
    class_names=['Unsurvived', 'Survived'],
    filled=True
)

import graphviz
f = open('titanic.dot')
dot_graph = f.read()
dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='titanic_tree')
print(dot)


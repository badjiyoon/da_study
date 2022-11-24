#CH05_04. 다양한 머신러닝 기법

#데이터 생성
import seaborn as sns
import pandas as pd

df = sns.load_dataset('titanic')

train_df = df[:800]
test_df = df[800:]

names = train_df.columns
train_df = train_df.drop(names[4:], axis = 1)
test_df = test_df.drop(names[4:], axis = 1)

train_df.fillna(train_df.mean()[['age']], inplace = True)
test_df.fillna(test_df.mean()[['age']], inplace = True)

map_dict = {'female' : 0, 'male' : 1}

train_df['sex'] = train_df['sex'].map(map_dict).astype(int)
test_df['sex'] = test_df['sex'].map(map_dict).astype(int)

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

X_train = train_df.drop(["survived"], axis=1)
Y_train = train_df["survived"]
X_test  = test_df.drop("survived", axis=1)
Y_test = test_df["survived"]

#결정나무
from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)

print(decision_tree.score(X_train, Y_train))
print(decision_tree.score(X_test, Y_test))

#배깅(랜덤 포레스트)
from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=100) #모델이 100개 정도
random_forest.fit(X_train, Y_train)
print(random_forest.score(X_train, Y_train))
print(random_forest.score(X_test, Y_test))
'''
데이터 하나에 모델이 여러 개, 하나 하나가 다 결정나무
데이터를 다양하게 뽑아서 여러 모델을 만들고, 
거기서 나온 값들로 최종 모델을 생성
'''

#부스팅(xgboost)
import xgboost as xgb
boosting_model = xgb.XGBClassifier(n_estimators = 100)
boosting_model.fit(X_train, Y_train)
print(boosting_model.score(X_train, Y_train))
print(boosting_model.score(X_test, Y_test))
'''
데이터를 가지고 모델을 만들고, 
단점을 보완하여 새로운 모델, 또 새로운 모델을 계속 만들어냄.
결점을 보완하여 모델들을 가지고 새로운 최종 모델 생성
'''
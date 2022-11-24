from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators = 100) 
random_forest.fit(X_train, Y_train)
print(random_forest.score(X_train, Y_train))
print(random_forest.score(X_test, Y_test))

import xgboost as xgb
boosting_model = xgb.XGBClassifier(n_estimators = 100)
boosting_model.fit(X_train, Y_train) # 학습
print(boosting_model.score(X_train, Y_train))
print(boosting_model.score(X_test, Y_test))
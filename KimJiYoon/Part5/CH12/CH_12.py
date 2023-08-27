#  [*.dat] : https://archive.ics.uci.edu/ml/machine-learning-databases/00224/
#  Part4. [실습10] 제조 공정 내 가스 혼합물의 개별 가스 성분 분류
from matplotlib import pyplot as plt

plt.rc('font', family='AppleGothic')
# ## 01. 데이터 소개 및 분석프로세스 수립
#  : "강의자료 → Ch11. [실습9] 가스 터빈 추진 플랜트 제조 공정의 부식 예측" 참고
# 02. 데이터 준비를 위한 EDA 및 전처리

# ### 0. 데이터 불러오기

############################################## 00. 필요한 파이썬 라이브러리 불러오기 #####################################################
import os
import time
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 데이터 파일을 얻어오기 위한 처리
path = '/Users/jiyoonkim/Documents/da_study/comFiles/Dataset'
all_files = glob.glob(os.path.join(path, "*.dat"))

df_from_each_file = (pd.read_csv(f, sep="\s+", index_col=0, header=None) for f in all_files)
df_from_each_file
df = pd.concat(df_from_each_file, sort=True)

# 데이터 모양의 변경 처리
# 각 셀마다 feature 와 value 를 나눠준다 (예 --> 1:15596.16 ---> 15596.16)
for col in df.columns.values:
    df[col] = df[col].apply(lambda x: float(str(x).split(':')[1]))

df = df.rename_axis('Gas').reset_index()
df.sort_values(by=['Gas'], inplace=True)
df.reset_index(drop=True, inplace=True)
# 1. 데이터 탐색
# 1) Basic
# 가스는 1의 6까지의 종류로 되어있음
df.Gas.nunique()
df.head()
df.shape
# 2) 데이터 타입
# object 컬럼 제외 -> Gas 컬럼
pd.unique(df.dtypes), len(df.select_dtypes(exclude='object').columns) - 1
# 3) 데이터 통계값
df.describe()
# 2. Feature 정의
# 1) Target Feature

# 1. Ethanol
# 2. Ethylene
# 3. Ammonia
# 4. Acetaldehyde
# 5. Acetone
# 6. Toluene
sns.countplot(df.Gas)
sns.set(style="darkgrid")
plt.title('Gas Count')
plt.show()

# 분포를 그려봄
sns.distplot(df.Gas)
plt.xlim(1, 6)
plt.title('Distribution of Gas')
plt.show()

# 2) 농도가 다른 컬럼 확인
conc = df.iloc[:, 1]
# 농도가 다른 것들을 비교하기 위해 처리
conc_red = conc.apply(lambda x: x / 10000)

fig = plt.figure(figsize=(22, 5))
fig.add_subplot(121)
sns.distplot(conc_red)
plt.title('Distribution of Concentrations')
plt.xlabel('Gas concentration Levels')

fig.add_subplot(122)
sns.boxplot(conc_red)
plt.title('Concentration')
plt.xlabel('Gas concentration Levels')

plt.show()

# #### 3) 데이터 확인
attr = df.iloc[:, 2:].copy()
attr.head()

# #### 4) 상관도 분석
# 상관계수 계산
correlation = df.corr()

# Heatmap 그리기
f, ax = plt.subplots(figsize=(20, 10))
plt.title('Correlations in dataset', size=20)
sns.heatmap(correlation)
plt.show()

# 상관계수 상위 20개 (양수, 음수)
# 상관계수 정렬
conc_corr = correlation.iloc[:, 1].sort_values(ascending=False)

# 상위 20개 (양수)
conc_corr[1:].head(20)

# 상위 20개 (음수)
conc_corr[1:].tail(20)

# 5) 상관도 기준 관계 그래프
fig = plt.figure(figsize=(20, 50))
for i in range(0, 20):
    fig.add_subplot(10, 2, i + 1)
    sns.scatterplot(attr.iloc[:, conc_corr.index[i]], conc_red, hue="Gas", palette="Set1", data=df, legend="full")
    plt.xlabel(conc_corr.index[i])
    plt.ylabel("Gas Concentration")

fig.tight_layout()
plt.show()

# PCA 적용
# 1) 데이터 준비
df_copy = df.copy()

X = df_copy.iloc[:, 1:]
# 각 가스들을 나타내는 숫자
y = df_copy.iloc[:, 0]
y.head()
X.head()

# 2) 테스트 모델 생성 주성분 3개로 표현
pca = PCA(n_components=3)
X_train = pca.fit_transform(X)
# 3) 그래프
# 평면 그래프 -> 3차원 그래프
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 11
# 가스별로 그려준다
ax.plot(X_train[0:2564, 0], X_train[0:2564, 1], X_train[0:2564, 2], 'o', markersize=2.5, label='Ethanol')
ax.plot(X_train[2565:5490, 0], X_train[2565:5490, 1], X_train[2565:5490, 2], 'o', markersize=2.5, label='Ethylene')
ax.plot(X_train[5491:7131, 0], X_train[5491:7131, 1], X_train[5491:7131, 2], 'o', markersize=2.5, label='Ammonia')
ax.plot(X_train[7132:9067, 0], X_train[7132:9067, 1], X_train[7132:9067, 2], 'o', markersize=2.5, label='Acetaldehyde')
ax.plot(X_train[9068:12076, 0], X_train[9068:12076, 1], X_train[9068:12076, 2], 'o', markersize=2.5, label='Acetone')
ax.plot(X_train[12077:13909, 0], X_train[12077:13909, 1], X_train[12077:13909, 2], 'o', markersize=2.5, label='Toluene')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(loc='upper right')

plt.show()

# t-SNE 그래프
# n_iter 최소 250개이상 -> 3000개
tsne = TSNE(n_components=3, n_iter=250)
xtrain = tsne.fit_transform(X)

# 각각의 평명산 표현하는 점을 보기 위한것임
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 11
ax.plot(xtrain[0:2564, 0], xtrain[0:2564, 1], xtrain[0:2564, 2], 'o', markersize=2.5, label='Ethanol')
ax.plot(xtrain[2565:5490, 0], xtrain[2565:5490, 1], xtrain[2565:5490, 2], 'o', markersize=2.5, label='Ethylene')
ax.plot(xtrain[5491:7131, 0], xtrain[5491:7131, 1], xtrain[5491:7131, 2], 'o', markersize=2.5, label='Ammonia')
ax.plot(xtrain[7132:9067, 0], xtrain[7132:9067, 1], xtrain[7132:9067, 2], 'o', markersize=2.5, label='Acetaldehyde')
ax.plot(xtrain[9068:12076, 0], xtrain[9068:12076, 1], xtrain[9068:12076, 2], 'o', markersize=2.5, label='Acetone')
ax.plot(xtrain[12077:13909, 0], xtrain[12077:13909, 1], xtrain[12077:13909, 2], 'o', markersize=2.5, label='Toluene')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.legend(loc='upper right')
plt.show()

# 4) Scailing
X_scaled = X.copy()
X_scaled = StandardScaler().fit(X_scaled).transform(X_scaled)

# 5) PCA Step 1 - Covariance Matrix 만들기
cov_matrix = np.cov(X_scaled.T)

# 6) PCA Step 2 - Eigen Values 와 Eigen Vector 만들기
eig_val, eig_vec = np.linalg.eig(cov_matrix)
print('Eigenvectors \n%s' % eig_vec)
print('\nEigenvalues \n%s' % eig_val)

tot = sum(eig_val)
var_exp = [(i / tot) * 100 for i in sorted(eig_val, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
print("누적 분산 설명력", cum_var_exp)

plt.plot(var_exp)
plt.show()

# %로 보면됨
plt.figure(figsize=(20, 4))
plt.bar(range(128), var_exp)
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.xlim(0, 20)
plt.xticks(range(-1, 20))
plt.tight_layout()
plt.show()

# 7) Scikit-learn 으로 PCA 적용하기
pca = PCA()
X_scaled = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 4))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0, 27)
plt.xticks(range(0, 27))
plt.title('Cumulative variance of principle components')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.tight_layout()
plt.show()

print(pca.explained_variance_ratio_)

# -> 12개의 성분만으로 약 95%의 설명력을 가질 수 있음
# 다양한 Classifier 를 활용한 가스 성분 분류 모델 생성
# 모델링 준비
# Label Encoding
from sklearn.preprocessing import label_binarize

y_ohe = label_binarize(y, classes=[1, 2, 3, 4, 5, 6])
n_classes = y_ohe.shape[1]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

y_train_nobinary = y_train.copy()
y_test_nobinary = y_test.copy()

# Scailing
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train = label_binarize(y_train, classes=[1, 2, 3, 4, 5, 6])
y_test = label_binarize(y_test, classes=[1, 2, 3, 4, 5, 6])


# ROC Curve Function 생성 -> 여러가지로 사용할 수 있도록 함수로 표현
def plot_roc(y_test, y_pred, title):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes
    lw = 2
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.3f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


# Confusion Plot Function 생성
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# 기본 모델링
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

clf = RandomForestClassifier()
random = clf.fit(X_train, y_train)

y_pred = random.predict(X_test)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('cross validation acc   :', cross_val_score(random, X_test, y_test).mean())

# > Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
class_names = ['1', '2', '3', '4', '5', '6']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Ada Boost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

ada = AdaBoostClassifier(n_estimators=10)
ada = ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)

# > Confusion Matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

plt.figure()
class_names = ['1', '2', '3', '4', '5', '6']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Bagging with KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier

bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=14),
                            max_samples=0.5, max_features=0.5)

bagging = bagging.fit(X_train, y_train)

y_pred = bagging.predict(X_test)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('cross validation acc   :', cross_val_score(bagging, X_test, y_test).mean())
# > Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['1', '2', '3', '4', '5', '6']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
tree = clf.fit(X_train, y_train)

y_pred = tree.predict(X_test)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('cross validation acc   :', cross_val_score(tree, X_test, y_test).mean())
# > Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['1', '2', '3', '4', '5', '6']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Majority Voting Ensemble Machine
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

clf1 = LogisticRegression()
clf2 = DecisionTreeClassifier()
clf3 = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
                 intercept_scaling=1, loss='squared_hinge', max_iter=1000,
                 multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
                 verbose=0)
clf4 = KNeighborsClassifier(n_neighbors=30)
eclf = VotingClassifier(estimators=[('lr', clf1), ('dt', clf2), ('svm', clf3), ('knn', clf4)], voting='hard')
eclf = eclf.fit(X_train, y_train)
y_pred = eclf.predict(X_test.values)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('ground truth           :', y_test)
print('predicted class        :', y_pred)
print('cross validation acc   :', cross_val_score(eclf, X_test, y_test).mean())

# -> Confusion Matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
class_names = ['1', '2', '3', '4', '5', '6']
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.show()
# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Confusion matrix, with normalization')
plt.legend(['train', 'test'], loc='lower right')
plt.show()

# Logistic Regression
# OneVsRestClassifier 는 클래스마다 분류기를 하나씩 만들어서 학습시키는 Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from scipy import interp
from itertools import cycle

start = time.time()

# Learn to predict each class against the other
classifier = OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1))
classifier.fit(X_train, y_train)
y_pred1 = classifier.predict_proba(X_test)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end - start))

# > Confusion Matrix
confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), y_pred1.argmax(axis=1))
confusion_matrix

# auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred1, axis=1))
auc_roc = metrics.classification_report(y_test.astype(int).tolist(), np.argmax(y_pred1, axis=1))

print('Logistic Regression Classification Report:\n {}'.format(auc_roc))
# plot_roc(y_test, y_pred1, "ROC Logistic Regression")

# SVC

from sklearn.svm import SVC

start = time.time()
classifier = OneVsRestClassifier(SVC(kernel="linear", verbose=1, decision_function_shape='ovr', probability=True))
classifier.fit(X_train, y_train)
y_pred2 = classifier.predict_proba(X_test)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end - start))

# > Confusion Matrix

# confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred2.argmax(axis=1))
confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), y_pred2.argmax(axis=1))
confusion_matrix

# auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred2, axis=1))
auc_roc = metrics.classification_report(y_test.astype(int).tolist(), np.argmax(y_pred2, axis=1))
print('SVC Classification Report:\n {}'.format(auc_roc))

# **We will use the macro average method to evaluate the algorithm.**

# Feature Selction 을 활용한 모델링

# * ROC(Receiver Operating Characteristic) : 모든 임계값에서 분류 모델의 성능을 보여주는 그래프
# * AUC(Area Under the Curve) : ROC 곡선 아래 영역을 의미함

# Logistic regression with RFE

from sklearn.feature_selection import RFE

start = time.time()

classifier = OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1))
rfe = RFE(classifier, n_features_to_select=64, verbose=1, step=1)
rfe = rfe.fit(X_train, y_train_nobinary)

end = time.time()
print("\nTime taken: {:.2f} seconds".format(end - start))

# RFE 알고리즘에 의해 선택된 Feature 리스트
rfe
features = X.columns[rfe.support_]
print(features)
X_train_rfe = pd.DataFrame(X_train)[features]
X_test_rfe = pd.DataFrame(X_test)[features]

classifier = OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1))
classifier.fit(X_train_rfe, y_train)
y_pred11 = classifier.predict_proba(X_test_rfe)

# > Confusion Matrix

confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred11.argmax(axis=1))
confusion_matrix

auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred11, axis=1))
print('Logistic regression with Recursive Feature Elimination:\n {}'.format(auc_roc))

plot_roc(y_test, y_pred11, 'ROC for Logistic regression with Recursive Feature Elimination')

# #### Logistic regression with SelectKBest (Chi Square test)
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest

# MinMaxScaler 적용
norm = MinMaxScaler()
X_train_norm = norm.fit_transform(X_train)

selector = SelectKBest(chi2, k=64)
selector.fit(X_train_norm, y_train)
X_train_kbest = selector.transform(X_train)
X_test_kbest = selector.transform(X_test)

classifier = OneVsRestClassifier(LogisticRegression(solver='sag', n_jobs=-1))
classifier.fit(X_train_kbest, y_train)
y_pred12 = classifier.predict_proba(X_test_kbest)

# > Confusion Matrix
# confusion_matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred12.argmax(axis=1))
confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), y_pred12.argmax(axis=1))
confusion_matrix

# auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred12, axis=1))
auc_roc = metrics.classification_report(y_test.astype(int).tolist(), np.argmax(y_pred12, axis=1))
print('Logistic regression with chi2 test feature selection:\n {}'.format(auc_roc))

plot_roc(y_test, y_pred12, 'ROC Logistic regression with chi2 test')

# #### SVC with RFE

classifier = OneVsRestClassifier(SVC(kernel="linear", decision_function_shape='ovr'))
rfe = RFE(classifier, n_features_to_select=64, verbose=1, step=1)
rfe = rfe.fit(X_train, y_train_nobinary)

features = pd.DataFrame(X_train).columns[rfe.support_]
print(features)
X_train_rfe = pd.DataFrame(X_train)[features]
X_test_rfe = pd.DataFrame(X_test)[features]

classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True, verbose=1, decision_function_shape='ovr'))
classifier.fit(X_train_rfe, y_train)
y_pred21 = classifier.predict_proba(X_test_rfe)

# confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred21, axis=1))
confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), y_pred12.argmax(axis=1))

# confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred21, axis=1))
confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), np.argmax(y_pred21, axis=1))
confusion_matrix

auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred21, axis=1))
print('SVC with Recursive Feature Elimination:\n {}'.format(auc_roc))

plot_roc(y_test, y_pred21, 'ROC for SVC with Recursive Feature Elimination')

# #### SVC with SelectKBest(chi2 test)

classifier = OneVsRestClassifier(SVC(kernel="linear", probability=True, verbose=1, decision_function_shape='ovr'))
classifier.fit(X_train_kbest, y_train)
y_pred22 = classifier.predict_proba(X_test_kbest)

# confusion_matrix = metrics.confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_pred22, axis=1))

confusion_matrix = metrics.confusion_matrix(y_test.astype(int).tolist(), y_pred22.argmax(axis=1))
confusion_matrix

auc_roc = metrics.classification_report(y_test.astype(int).tolist(), np.argmax(y_pred22, axis=1))
# auc_roc = metrics.classification_report(np.argmax(y_test, axis=1), np.argmax(y_pred22, axis=1))
print('SVC with chi2 test feature selection:\n {}'.format(auc_roc))

plot_roc(y_test, y_pred22, 'ROC for SVC with feature selection based on chi2 test')

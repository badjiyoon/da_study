import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(2021)

from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
target = iris.target

data = data[target !=0, :2]
target = target[target !=0]

plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()

from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.9, random_state=2021
)

from sklearn.svm import SVC


linear_svc = SVC(kernel="linear")
linear_svc.fit(train_data, train_target)


def plot_support_vector_machine(svm):
    # 전체 데이터
    plt.scatter(data[:, 0], data[:, 1], c=target, zorder=10, cmap=plt.cm.Paired,
                edgecolor='k', s=20)
    # test 데이터
    plt.scatter(test_data[:, 0], test_data[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = data[:, 0].min()
    x_max = data[:, 0].max()
    y_min = data[:, 1].min()
    y_max = data[:, 1].max()

    # 영역 칠하기
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = svm.decision_function(np.c_[XX.ravel(), YY.ravel()])

    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired, shading="auto")
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])


plt.figure(figsize=(7,7))
plot_support_vector_machine(linear_svc)

poly_svc = SVC(kernel="poly")
poly_svc.fit(train_data, train_target)

plt.figure(figsize=(7,7))
plot_support_vector_machine(poly_svc)

poly_svc = SVC(kernel="poly", gamma=0.1)
poly_svc.fit(train_data, train_target)

plt.figure(figsize=(7,7))
plot_support_vector_machine(poly_svc)

poly_svc = SVC(kernel="poly", gamma=10)
poly_svc.fit(train_data, train_target)

plt.figure(figsize=(7,7))
plot_support_vector_machine(poly_svc)




poly_svc = SVC(kernel="poly", gamma=10, degree=2)
poly_svc.fit(train_data, train_target)

plt.figure(figsize=(7,7))
plot_support_vector_machine(poly_svc)

poly_svc = SVC(kernel="poly", gamma=10, degree=4)
poly_svc.fit(train_data, train_target)

plt.figure(figsize=(7,7))
plot_support_vector_machine(poly_svc)

rbf_svc = SVC(kernel="rbf")
rbf_svc.fit(train_data, train_target)

plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)

rbf_svc = SVC(kernel="rbf", gamma=0.1)
rbf_svc.fit(train_data, train_target)

plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)

rbf_svc = SVC(kernel="rbf", gamma=10)
rbf_svc.fit(train_data, train_target)

plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)

poly_svc = SVC(kernel="poly", gamma=10)
poly_svc.fit(train_data, train_target)

hard_penalty_poly_svc = SVC(kernel="poly", gamma=10, C=100)
hard_penalty_poly_svc.fit(train_data, train_target)

plt.figure(figsize=(14, 7))
plt.subplot(1,2,1)
plot_support_vector_machine(poly_svc)
plt.title("No penalty")
plt.subplot(1,2,2)
plot_support_vector_machine(hard_penalty_poly_svc)
plt.title("Hard penalty")


rbf_svc = SVC(kernel="rbf", gamma=10)
rbf_svc.fit(train_data, train_target)

hard_penalty_svc = SVC(kernel="rbf", gamma=10, C=100)
hard_penalty_svc.fit(train_data, train_target)

plt.figure(figsize=(14, 7))
plt.subplot(1,2,1)
plot_support_vector_machine(rbf_svc)
plt.title("No penalty")
plt.subplot(1,2,2)
plot_support_vector_machine(hard_penalty_svc)
plt.title("Hard penalty")








import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(2021)

# 1. Data
from sklearn.datasets import load_iris

iris = load_iris()

data = iris.data
target = iris.target

data = data[target != 0, :2]
target = target[target != 0]

plt.scatter(data[:, 0], data[:, 1], c=target)
plt.show()

# 2.Linear Kernel
from sklearn.model_selection import train_test_split

train_data, test_data, train_target, test_target = train_test_split(
    data, target, train_size=0.9, random_state=2021
)

from sklearn.svm import SVC

lenear_svc = SVC(kernel="linear")
lenear_svc.fit(train_data, train_target)


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


# 아래 plot의 속성들은 다음과 같습니다.
# - decision boundary
#     - 검은색 직선
# - support vector
#     - 검은색 점선
# - 영역
#     - 클래스로 구별되는 영역
plt.figure(figsize=(7, 7))
plot_support_vector_machine(lenear_svc)
plt.show()

# 3.Poly Kernel
# 다음으로 알아볼 커널은 poly 커널 입니다.
# poly커널은 직선을 곡선으로 mapping 시켜주는 커널입니다.
# poly커널에 영향을 미치는 argument들은 다음과 같습니다.
# - gamma
#     - 결경 경계를 스케일링해주는 값입니다.
# - degree
#     - 몇 차원의 곡선으로 맵핑할지 정해주는 값입니다.
# 3.1 gamma
# 3.1.1 "scale"
# default 옵션은 자동으로 scaling 해줍니다.
poly_svc = SVC(kernel="poly")
poly_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(poly_svc)
plt.show()

# 3.1.3 gamma=10
# gamma 값 크게 맞추기
poly_svc = SVC(kernel="poly", gamma=10)
poly_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(poly_svc)
plt.show()

# 3.2 degree
# 3.2.1 degree=2
poly_svc = SVC(kernel="poly", gamma=10, degree=2)
poly_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(poly_svc)
plt.show()

# 3.2.1 degree=4
poly_svc = SVC(kernel="poly", gamma=10, degree=4)
poly_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(poly_svc)
plt.show()

# 4. RBF Kernel
# 다음으로 알아볼 것은 rbf커널 입니다.
# rbf 커널은 데이터를 고차원의 공간으로 mapping시켜줍니다.
# rbf또한 gamma 값으로 scaling을 합니다.
# 4.1 Scale
rbf_svc = SVC(kernel="rbf")
rbf_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)
plt.show()

# 4.2 gamma=0.1
rbf_svc = SVC(kernel="rbf", gamma=0.1)
rbf_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)
plt.show()

# 4.2 gamma=10
rbf_svc = SVC(kernel="rbf", gamma=10)
rbf_svc.fit(train_data, train_target)
plt.figure(figsize=(7, 7))
plot_support_vector_machine(rbf_svc)
plt.show()

# 5. Penalty
# 패널티는 `C` argument를 통해 줄 수 있습니다.

# 5.1 Poly
poly_svc = SVC(kernel="poly", gamma=10)
poly_svc.fit(train_data, train_target)
hard_penalty_poly_svc = SVC(kernel="poly", gamma=10, C=100)
hard_penalty_poly_svc.fit(train_data, train_target)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plot_support_vector_machine(poly_svc)
plt.title("No penalty")
plt.subplot(1, 2, 2)
plot_support_vector_machine(hard_penalty_poly_svc)
plt.title("Hard penalty")
plt.show()

# 5.2 RBF
rbf_svc = SVC(kernel="rbf", gamma=10)
rbf_svc.fit(train_data, train_target)
hard_penalty_svc = SVC(kernel="rbf", gamma=10, C=100)
hard_penalty_svc.fit(train_data, train_target)
plt.figure(figsize=(14, 7))
plt.subplot(1, 2, 1)
plot_support_vector_machine(rbf_svc)
plt.title("No penalty")
plt.subplot(1, 2, 2)
plot_support_vector_machine(hard_penalty_svc)
plt.title("Hard penalty")
plt.show()

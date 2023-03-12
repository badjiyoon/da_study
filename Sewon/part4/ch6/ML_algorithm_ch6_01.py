#CH06_01. KNN

"""
모델 학습 방법
1. 모델 기반 학습(Model-Based Learning)
-데이터로부터 모델을 생성하여 분류/예측 진행
-예) Linear Regression, Logistic Regression
  X new -> f(X) -> Y new

2. 사례 기반 학습(Instance-Based Learning)
-별도의 모델 생성 없이 인접 데이터를 분류/예측에 사용
-Lazy Learning: 모델을 미리 만들지 않고, 새로운 데이터가 들어오면 계산을 시작
-예) KNN, Naive Bayes\
 X new -> Find Nearest X s -> Y new
 
K-Nearest Neighbors
-K개의 가까운 이웃을 찾는다.
-학습 데이터 중 K개의 가장 가까운 사례를 사요하여 분류 및 수치 예측

Step1) 새로운 데이터를 입력 받음
Step2) 모든 데이터들과의 거리를 계산
Step3) 가장 가까운 K개의 데이터를 선택
Step4) K개 데이터의 클래스를 확인
Step5) 다수의 클래스를 새로운 데이터의 클래스로 예측

K값에 따른 결정 경계(Decision Boundary)
-이웃을 적게 사용(Overfitting) <-> 이웃을 많이 사용(Underfitting)

Cross Validation
-교차 검증을 통해서 제일 성능이 좋은 K를 선택
 ->예를 들어 1~10 사이의 K중 제일 성능이 좋은 K를 선택

*K는 홀수로 지정 -> 짝수의 경우 동점이 발생할 수 있기 때문

거리의 종류
1. 유클리드 거리(Euclidean Distance)
-두 점 사이의 거리를 계산할 대 흔히 쓰는 방법
-두 점 사이의 최단거리를 의미

2. 맨해튼 거리(Manhattan Distance)
-한 번에 한 축 방향으로 움직일 수 있을 때 두 점 사이의 거리

K-Nearest Neighbors 장점
-학습 과정이 없다.
-결과를 이해하기 쉽다.

K-Nearest Neighbors 단점
-데이터가 많을 수록 시간이 오래 걸린다.
-지나치게 데이터에 의존적이다.
"""
#CH07_01. SVM

"""
Support Vector Machine(SVM)
-SVM은 결정 경계(Decision Boundary)를 정의하는 모델

Support Vector Machine의 목표
-Margin을 최대화하는 Decision Boundary(결정 결계) 찾기
-이 때 Support Vector가 Decision Boundary를 만드는데 영향을 주기 때문에 Support Vector Machine이라고 불림
 *Margin: 여백, 간격

Support Vector Machine의 구성 요소
-Support Vector: 두 클래스 사이의 경계에 위치한 데이터 포인트들
-Margin: Decision Boundary와 Support Vector 사이의 거리 X2
-Decision Boundary: 데이터를 나누는 기준이 되는 경계
 *hyperplane: R2 -> line, R3 -> plane

*Hard Margin vs Soft Margin
Hard Margin
-어떠한 오분류도 허용하지 않음

Soft Margin
-어느 정도의 오분류는 허용 -> Penalty
-Penalty의 종류
 1) 0-1 Loss (제로원 로스)
  -Error가 발생한 개수만큼 패널티 계산

 2) Hinge Loss
  -오분류 정도에 따라 패널티 계산

Q.Hard Margin일 경우 어떻게 나누어야 할까? 
 -> Non linear SVM: 데이터를 나누는 결정 경계를 찾는다.
 -> 직선으로 되어 있는 데이터를 곡선으로 Mapping

*커널 트릭: 저차원 데이터를 고차원 데이터로 맵핑(Mapping)하는 작업

커널 종류
-선형
-다항식
-가우시안 RBF(Radial Basis Function)
 : 이 함수는 벡터 l이라는 랜드마크와 벡터 x가 얼마나 가까운지에 따라 0에서 1사이의 값을 보이는데, 
   가장 가까울 때 1, 멀 때 0의 값을 가진다. 

SVM 장단점
장점
-비선형 분리 데이터도 커널을 사용하여 분류할 수 있다.
-고차원 데이터에도 사용할 수 있다.

단점
-데이터가 너무 많으면 속도가 느리다.
-확률 추정치를 제공하지 않는다.

"""
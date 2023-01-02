# CH01_02. Model Selection

'''
1. 모델
-정의: 어떤 X가 주어졌을 때 f라는 함수를 통해 y라는 값을 도출하는 과정
이 때의 함수 f를 모델 또는 알고리즘이라고 부른다.

-모델의 수식
y=F(X) (X: 데이터, y: 예측값)
-모델의 목적: 데이터를 이용해 값을 예측
-모델의 평가: 모델이 값을 잘 예측하는지 평가

2. 데이터의 종류
-Train data: 학습에 사용되는 데이터
-Test data: 학습에 사용되지 않은 데이터
모델이 실제로 잘 예측하는지 알기 위해서는 하긋ㅂ에 사용되지 않은 데이터를 이용해 평가해야 한다.

3. 모델의 평가와 데이터의 관계
-과소적합(Underfitting): 학습데이터는 잘 맞추지만 학습데이터 외에는 잘 맞추지 못하는 현상
-과대적합(Overfitting): 학습데이터를 잘 맞추지 못하는 현상

*Underfitting 확인하는 방법
-Train data로 학습된 모델을 Train data로 평가한다.
-Train data를 잘 맞추지 못한다면 Underfitting 상태

*Overfitting 확인하는 방법
-Train data를 잘 학습한 모델을 Test data로 평가한다.
-Traing data는 잘 맞추지만 Test data를 잘 맞추지 못한다면 Overfitting 상태

-Data split: 데이터를 Train data와 Test data로 나누는 것 (보통 7:3)
-각 데이터의 용도
Train data: 학습에 사용되는 데이터
Valid data: 학습이 완료된 모델을 검증하기 위한 데이터, 학습에 사용되지는 않지만 관여하는 데이터
Test data: 최종 모델의 성능을 검증하기 위한 데이터, 학습에도 사용되지 않으며 관여하지도 않는 데이터

*문제점
Overfitting: valid 데이터에 Overfitting될 수 있음

4. Cross Validation(교차검증)
-정의: valid 데이터를 고정하지 않고 계속해서 변경함으로써 Overfitting 되는 것을 막기 위한 방법

1)LOOCV (Leave One Out Cross Validation)
-방법: 하나의 데이터를 제외하고 모델을 학습한 후 평가
-문제점: 데이터 개수만큼의 모델을 학습해야 하므로 데이터가 많으면 시간이 오래 걸린다.

2) K-Fold
-방법: 데이터를 K개로 분할한 후 한 개의 분할 데이터를 제외한 후 학습에 사용
제외된 데이터는 학습이 완료된 후 평가에 사용

-Cross Validation 평가
1) Cross Validation을 이용하면 방법에 따라 K개의 평가지표(LOOCV는 n개)가 생성
2) 생성된 평가 지표의 평균을 이용해 모델의 성능을 평가
3) 전체 Train 데이터를 이용해 모델 학습
'''
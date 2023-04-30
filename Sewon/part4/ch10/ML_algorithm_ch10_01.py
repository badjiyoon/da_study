#CH10_01. Boosting
"""
1. Boosting
Boosting의 정의
-약 분류기를 순차적(Sequential)으로 학습하는 앙상블 기법
-예측을 반복하면서 잘못 예측한 데이터에 가중치를 부여해서 오류를 개선한다.

Bagging
-무작위 복원 추출로 부트스트랩 샘플을 추출한다.

Boosting
-저부트스트랩 샘플을 추출하는 과정에서 각 자료에 동일한 확률을 부여하는 것이 아니라
분류가 잘못된 데이터에 더 큰 가중을 주어 표본을 추출한다

Boosting의 종류
-AdaBoost
-Gradient Boost
-XGBoost

2. AdaBoost
-Adaptive Boosting
-간단한 약 분류기들이 상호 보완하도록 순차적으로 학습한다.
-과소적합된 학습 데이터의 가중치를 높이면서(Adaptive) 
 새로 학습된 모델이 학습하기 어려운 데이터에 더 잘 적합되도록 하는 방식

약 분류기
-변수 하나와 if문 하나 정도의 depth

AdaBoost 학습 방법
Step1) 전체 학습 데이터를 이용해 모델을 생성
Step2) 잘못 예측된 데이터의 가중치를 상대적으로 높여줌
Step3) 가중치를 반영하여 다음 모델을 학습
Step4) Step2~3 과정을 반복한다
*신뢰도=(1/2)*ln{(1-e)/e}
에러율=(오류 데이터 가중치의 합)/(전체 데이터 가중치의 합)

AdaBoost 예측 방법
-각 모델의 신뢰도를 곱하여 Voting

3. Gradient Boost
-학습 전 단계 모델에서의 잔여 오차(Residual error)에 대해 새로운 모델을 학습
-잔여 오차를 예측하여 발전하는 약분류기
-현재까지 학습된 분류기의 약점을 Gradient를 통해 알려주고 이를 중점으로 보완하는 방식

Gradient Boost 학습 방법
Step1) 모델을 학습 후 예측값을 계산한다.
Step2) 잔여 오차를 계산한다.
Step3) 잔여 오차를 예측하는 트리 모델 생성한다.
Step4) Learning Rate를 이용해 기존 예측값을 수정한다.
Step5) Step2~4를 반복한다

Gradient Boost 예측
-최초 모델의 예측값에 생성된 잔여 오차 예측 모델의 오차 예측값을 더한다.

4. XGBoost
-eXtreme Gradient Boosting
-Gradient Boosting 기반의 모델
-트리를 만들 때 병렬 처리를 가능하게 해서 Gradient Boosting의 속도를 개선

XGBoost의 특징
1) 병렬 / 분산 처리
-CPU 병렬 처리가 가능
-코어들이 각자 할당받은 변수들로 제각기 가지를 쳐 나감
2) Split 지점을 고려할 때 일부를 보고 결정
-연속형 변수들의 Split지점을 고려할 때 일부분만 보고 고려함
3) Sparsity Awareness
-Zero 데이터를 건너 뛰면서 학습
-범주형 변수를 dummy화 시킬 경우 학습 속도를 빠르게 할 수 있음
"""
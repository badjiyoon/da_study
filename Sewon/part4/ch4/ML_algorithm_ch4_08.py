#CH04_08. Ensemble & Random Forest

"""
1. Ensemble
-앙상블의 정의: 약한 분류기들을 결합하여 강 분류기로 만드는 것
-앙상블의 종류
 1) Bagging
 2) Boosting
 3) Stacking

2. Bagging
-Bagging = Bootstrap(스스로의 힘) + Aggregation(종합)

1) Bootstrap
-Train Data에서 여러 번 복원 추출하는 Random Sampling 기법
 추출된 샘플들을 부트스트랩 샘플이라고 부른다.
 이론적으로 36.8%의 샘플이 뽑히지 않게 됨(Out-Of-Bag 샘플)
-OOB(Out-Of-Bag) 평가
 추출되지 않는 샘플을 이용해 Cross Validation(교차 검증)에서 Validation data로 활용
-약분류기 생성
 추출된 부트스트랩 샘플마다 약분류기를 학습한다.

2) Aggregation
-생성된 약분류기들의 예측 결과를 Voting을 통해 결합한다.
-Aggregation의 종류: Hard Voting, Soft Voting
 Hard Voting: 예측한 결과값 중 다수의 분류기가 결정한 값을 최종 예측값으로 선정
 Soft Voting: 분류기가 예측한 확률값의 평균으로 결정

-Bagging의 장점
분산을 줄이는 효과
 -원래 추정 모델이 불안정하면 분산 감소 효과를 얻을 수 있음
 -과대 적합이 심한(High-Variance) 모델에 적합

3. Random Forest
-Random Forest = Decision Tree + Bagging
-분산이 큰 Decision Tree + 분산을 줄일 수 있는 Bagging

-Random Forest와 무작위성(Randomness)
 무작위성을 더 강조하여서 의사결정나무들이 서로 조금씩 다른 특성을 가짐
 (변수가 20개가 있다면 5개의 변수만 선택해서 의사결정나무를 생성)
 의사결정나무의 예측들이 비상관화되어 일반화 성능을 향상 (Overfitting 위험 감소)

-Random Forest 학습 방법
 step1) Bootstrap 방법으로 T개의 부트스트랩 샘플을 생성한다.
 step2) T개의 의사결정나무들을 만든다.
 step3) 의사결정나무 분류기들을 하나의 분류기로 결합한다. (Voting)

-Random Forest의 장단점
장점
-의사결정나무의 단점인 Overfitting을 해결
-노이즈 데이터의 영향을 크게 받지 않는다.
-의사결정나무 모델보다 복잡도가 적다.

단점
-모델의 예측 결과를 해석하고 이해하기 어렵다.
"""
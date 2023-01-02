#CH03_01. Logistic Regression

'''
1. Logistic Regression 정의
2. Area Under ROC (AUROC)
3. Multicalss Logistic Regression
'''

'''
1. Logistic Regression 정의
데이터의 형태
-연속형 데이터: Linear Regression
-범주형 데이터: 
 1) 정답의 범위가 0과 1 사이
 2) 단순 선형 회귀의 예측값의 범위는 -∞ ~ ∞
 3) 0과 1 사이를 벗어나는 예측은 예측의 정확도를 낮추게 됨
 4) 예측의 결과가 0과 1 사이에 있어야 한다.
 => Logistic Function: y=1/(1+e^(-x))

Logistic Regression의 정의
-Linear Regression + Logistic Regression
-정답이 범주형일 때 사용하는 Regression Model

Logistic Regression의 수식
-Linear Regression: y=b0+b1x
-Logistic Function: y=1/(1+e^(-x))
-Linear Regression+Logistic Regression: P(y=1)=1/(1+e^-(b0+b1x))

Threshold의 정의
-확률값을 범주형으로 변환할 때의 기준
 예) Threshold=0.5 -> 확률이 0.5보다 크면 1, 확률이 0.5보다 작으면 0

2. Area Under ROC (AUROC)
AUROC
-정확도는 Threshold에 따라 변하기 때문에 지표로서 부족할 때가 있다.
 이를 보완하기 위한 Threshold에 의해 값이 변하지 않는 지표가 AUROC

ROC Curve
-True Positive Ratio 대 False Positive Ratio 의 그래프
-Confusion Matrix: 실제 값과 예측 값을 매트릭스 형태로 표현한 것
-True Positive: 실제 1을 1이라고 예측한 수
-False Negative: 실제 1을 0이라고 예측한 수
-False Positive: 실제 0을 1이라고 예측한 수
-True Negative: 실제 0을 0이라고 예측한 수
-True Positive Ratio(TPR) = True Positive / (True Positive + False Negative)
-False Positive Ratio(FPR) = False Positive / (False Positive + True Negative)

AUROC = ROC Curve의 밑 면적의 넓이

Best Threshold => Youdens' Index
-J= TPR + True Negative / (True Negative + False Positive) - 1
-J를 가장 크게 하는 Threshold를 Best로 기준
 *추가 설명이 필요할듯

3. Multicalss Logistic Regression
One or Nothing
-범주가 3개일 때: A, B, C
 1) A or Not
 2) B or Not
 3) C or Not -> 각각 Logistic Regression
 P(A)=0.3, P(B)=0.5, P(C)=0.2 -> 합은 1
-확률이 제일 높은 class가 예측값이 된다.
'''

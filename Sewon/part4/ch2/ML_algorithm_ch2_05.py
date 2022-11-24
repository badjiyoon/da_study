# CH02_05. Regularization

'''
*Overfitting 방지하는 법
1) 더 많은 학습데이터
2) 모델의 정규화

-모델의 정규화: 
모델에 제한을 주어 학습데이터의 패턴을 모두 외우는 것을 방지하는 방법

*선형회귀 + 정규화
-loss의 최소화: argmin(y-BX)^2
-불필요한 beta는 학습하지 말자!: argmin(y-BX)^2 + 람다*베타 놈들의 합

베타 놈 = 계수들의 norm

L1 norm: 그냥 베타들의 절댓값의 합
L2 norm: 베타 제곱들의 합

1) LASSO (least absolute shrinkage and selection operator)
-L1 norm
-필요없는 베타를 0으로 만든다.

2) Ridge
-L2 norm
-필요없는 베타를 아주 작은 값으로 만든다.
'''
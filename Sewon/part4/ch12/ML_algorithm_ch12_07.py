#CH12_07. Collaborative Filtering (2)
"""
1. 모델 기반 (Model Based) 협업 필터링
-여러 유저의 과거 아이템 상호작용 정보를 이용해 추천을 위한 모델을 학습하고, 학습된 모델을 이용해 추천


2. Matrix Factorization
-유저‒아이템 행렬을 유저와 아이템 행렬로 분해하는 방법
-유저가 평가하지 않은 아이템에 대한 선호도를 예측 가능
-User-Item Matrix=User Latent Matrix P × Item Latent Matrix Q
 (N × M) = (N × K) · (K × M)
 * K=유저와 아이템 행렬이 공유하는 잠재 요인

-Gradient Descent를 이용한 Matrix Factorization 학습
 Step1) P와 Q를 랜덤값으로 초기화
 Step2) 𝑅 계산
 Step3) 𝑅과 𝑅의 오차 계산
 Step4) Gradient Descent를 이용해 P와 Q를 업데이트
 Step5) Step2~4 과정 반복
"""
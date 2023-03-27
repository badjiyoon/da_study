#CH08_01. Clustering
"""
1. 군집화(Clustering)의 정의
-유사한 속성을 갖는 데이터들을 묶어 전체 데이터를 몇 개의 군집으로 나누는 것

Classification
-Supervised Learning
-소속 집단의 정보를 알고 있는 상태
-Label이 있는 데이터를 나누는 방법

Clustering
-Unsupervised Learning
-소속 집단의 정보를 모르고 있는 상태
-Label이 없는 데이터를 나누는 방법

군집 분석의 종류
1) 계층적(Hierarchical) 군집화
2) 비계층적(Non-Hierarchical) 군집화

2. 계층적 군집화 (Hierarchical)
-개체들을 가까운 집단부터 묶어 나가는 방식
-유사한 개체들이 결합되는 dendrogram 생성
-Cluster들은 sub-cluster를 갖고 있다.

거리의종류
-유클리드 거리
-맨해튼 거리
-표준화 거리
-민콥스키 거리

Hierachical Clustering 종류
-묶인 클러스터와 다른 데이터간의 거리 측정 방법에 따라 달라진다.

1) 최단 연결법: 군집에서 가장 가까운 데이터가 새로운 거리
distance(UV)W= min(distanceUW, distanceVW)

2) 최장 연결법: 군집에서 가장 먼 데이터가 군집과 데이터의 거리
distance(UV)W= max(distanceUW, distanceVW)

3) 평균 연결법: 군집의 데이터들간의 거리의 평균이 군집과 데이터의 거리

4) 중심 연결법: 군집의 중심이 새로운 거리
distance(G1, G2) = x1과 x1의 유클리드 거리

주어진 Cluster의 개수에 맞게 데이터를 나누는 방법
-dendrogram에서 가로로 자른다.

3. 군집화 평가
좋은 Clustering이란?
-군집 내 유사도를 최대화 (거리를 최소화)
-군집 간 유사도를 최소화 (거리를 최대화)

내부 평가
-군집된 결과 그 자체를 놓고 평가하는 방식
 ex) Dunn Index, Silhouette 등

Dunn Index
DI=(군집과 군집 사이의 거리 중 최소값)/(군집 내 데이터간 거리 중 최대값)
-군집과 군집 사이의 거리가 클 수록, 군집 내 데이터간 거리가 작을 수록 좋은 모델
-> DI가 큰 모델

Silhouette Index
S={b(i)-a(i)}/max{a(i), b(i)}

-군집 내 응집도(cohesion)
a(i): 데이터와 동일한 군집 내의 나머지 데이터들과의 평균 거리

-군집 간 분리도(separation)
b(i): 데이터와 가장 가까운 군집 내의 모든 데이터들과의 평균 거리

외부 평가
-군집화에 사용되지 않는 데이터로 평가하는 방식
"""
# CH02_08. 자료형_셋
#셋은 수학의 집합과 비슷한 개념

#생성 방법
a=set([1, 2, 3])
print(a)
print(type(a))

a=set((1, 2, 3))
print(a)

a=set({1, 2, 3})
print(a)

#중복된 항목 제거
a=set((1, 1, 2, 3, 3, 4))
print(a)

#숫서가 없음 > 인덱싱 없음
a=set((4, 4, 3, 2, 1, 'a', 'b', 'a'))
print(a)
b=list(a)
print(b)
print(b[2])

#합집합
a=set((1, 2, 3))
b=set((3, 4, 5))
c=a.union(b) #합집합 함수 union
print(c)
d=a.intersection(b) #교집합 함수 intersection
print(d)

#차집합
a=set((1, 2, 3))
b=set((3, 4, 5))
c=a.difference(b) #차집합 함수 difference
print(c)

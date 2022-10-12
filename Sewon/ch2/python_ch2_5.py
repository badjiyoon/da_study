# CH02_05. 자료형_리스트

a1=1
a2=3
a3=5
a4=5
a5=2
a6=4 #괜히 썼네

a=[1, 3, 5, 5, 2, 4]
print(a)
print(type(a))

#리스트의 구조 /인덱스
b=[1, [1, 2], [1, [1, 2, 3]]]
print(b[0])
print(b[1][1])
print(b[2][1][0])
print(b[2][1][2])

#슬라이스
a=[1, [1, 2], [1, [1, 2, 3]]]
print(a[0])
print(a[-1])
print(a[1:])
print(a[2][1][1:])

#치환
a=[1, [2, 3], [4, [5, 6, 7]]]
a[0]=10
print(a)
a[2][1][2]=10
print(a)
print(a[2][1][2])
a[2][1][1:]=50
print(a[2][1][1:])
a[2][1][1:]=[50]
print(a[2][1][1:])
print(a)

#리스트 더하기
# CH02_05. 자료형_리스트

a=[1, 2, 3]
b=[4, 5, 6]
print(a+b)

#리스트 곱하기
a=[1, 2, 3]
print(a*2)

#append
a=[1, 2, 3]
a.append(4)
print(a)
a.append([5, 6])
print(a)

#pop
a=[1, 2, 3]
a.pop()
print(a)
a.pop(0)
print(a)
a.pop(0)
print(a)

#index
a=[1, 2, 3]
print(a.index(2))
print(a.index(3))
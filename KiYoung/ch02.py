

print(10 + 10)

a = 10
b = 100

print(a)
print(b)
print(a + b)

print(a * b)

print(a / b)

print(a + b)
print(a - b)
print(a * b)
print(a / b)


a = 100
A = 10
print(a)
print(A)


a = 10
b = 0.1
c = 'hey'

print(type(a))
print(type(b))
print(type(c))


a = 1
print(type(a))
b = 0
print(type(b))
c = -1
print(type(c))


a = 1.11
print(type(a))
b = 0.0
print(type(b))
c = -1.12
print(type(c))
d = 1.0
print(type(d))

a = 1
b = 1.0
print(type(a))
print(type(b))
print(type(a+b))

a = 40
b = 18
print(a + b)
print(a - b)
print(a * b)
print(a / b)
a = 10
b = 3
print(a // b)
print(a % b)
a ** b
a = 1
a = a + 1
print(a)
a = 1
a -= 1
print(a)
a = 2
a += 2
print(a)
a = 1
a /= 2
print(a)


a = 'hello world'
a = "hello world"
a = 'hello world'
print(a[0])
print(a[-1])

print(a)
print(a[:])
print(a[3:5])
print(a[:3])
print(a[:-4])
a = 'hello'
b = 'world'
print(a + b)
a = 'hello'
print(a * 3)
a = 'My name is jk'
b = 'aaa'
print(a, b)

print(a + str(b))
print('My number is {}'.format(b))
print('오늘은 {}월 {}일 입니다.'.format(5, 10))
print('오늘은 {month}월 {day}일 입니다. {month}월은 가을.'.format(month = 5, day = 10))

print(a)
a = 'hello world'
print(a.split())

# 리스트
a1 = 1
a2 = 3
a3 = 5
a4 = 5
a5 = 2
a6 = 4

a = [1, 3, 5, 5, 2, 4]
print(a)
print(type(a))

b = [1, [2, 3], [1, [5, 6]]]
print(b)
print(b[0])


a = [1, [2, 3], [4, [5, 6, 7]]]
print(a[0])
print(a[-1])
print(a[1:])

a[0] = 10
print(a)

a[2][1][1:] = [60, 70]
print(a)

print(a[2][1][1:])
a[2][1][1:] = [50]
print(a)

a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)
a = [1, 2, 3]
print(a * 2)

a = [1, 2, 3]
a.append(4)
print(a)
a = [1, 2, 3]
a.append([4, 5])
print(a)
a = [1, 2, 3]
a.pop()
print(a)
a = [1, 2, 3]
print(a.index(2))

a = (1, 2, 3)
print(type(a))
print(a[1])

a = (1, 2, 3)
print(a[:2])
a = (1, 2, 3)

a = (1, 2, 3)
b = (4, 5, 6)
c = a + b
print(c)
# 튜플 곱하기
a = (1, 2, 3)
b = a * 2
print(b)
a = {'a' : 'b', 'c' : 'd', 'e' : 'f'}
print(a)
print(type(a))

a = {'aa' : ['bb', 'cc', 'dd'], 'ee' : 'ff'}
print(a)
# key 얻기
key = a.keys()
print(key)
print(type(key))



a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


for i in a :
    print(i)


for i in range(1, 11) :
    print(i)


for i in range(1, 10, 2) :
    print(i)


for i in range(9, 0, -2) :
    print(i)


result = 0
for i in range(1, 11, 1) :
    result += i
    print(result)

for i in range(1, 10) :
    for j in range(1, 10) :
        print('{0} * {1} = {2}'.format(i, j, i*j))


i = 1
while (i < 11) : 
    print(i)
    i += 1


x = 6
if x > 5 :
    print(True)

x = 4
if x > 5 :
    print(True)
else :
    print(False)



x = 95
if x >= 90 :
    print('A')
elif x >= 80 :
    print('B')
elif x >= 70 :
    print('C')
elif x >= 60:
    print('D')
else : 
    print('F')
 

A = [80, 95, 70, 55, 63]
for i in A :
    print(i)

for i in A :
    if i >= 90 :
        print(i, 'A')
    elif i >= 80 :
        print(i, 'B')
    elif i >= 70 :
        print(i, 'C')
    elif i >= 60:
        print(i, 'D')
    else : 
        print(i, 'F')



# 함수
def function1(x) : 
    print(x + 1)

def function2(x) :
    result = x + 1
    return result

def function3(x) :
    result1 = x + 1
    result2 = x + 2
    return result1, result2

def function4(x):
    for i in x :
        print(i)



function1(1)
a = function2(1)
print(a)

a, b = function3(1)
print(a)
print(b)



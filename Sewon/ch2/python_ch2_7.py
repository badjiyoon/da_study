# CH02_07. 자료형_딕셔너리

#딕셔너리 생성 방법: key + value

a={'사자':'lion', '호랑이':'tiger', '용':'dragon'}
print(a)
print(type(a))

#하나의 key에 여러 value
a={'car':['bus', 'truck', 'taxi'], 'train':'ktx'}
print(a)

#key 얻기
a={'car':['bus', 'truck', 'taxi'], 
    'train':'ktx'}
print(a)

key=a.keys()
print(key)
print(type(key)) #'dict_keys'
print(key[0]) #딕트는 인덱스로 값 못 빼네

key2=list(key) #타입 변환: dict_keys > list
print(type(key2))
print(key2[1]) #리스트로 변환됐으니 인덱싱 가능
print(key2)

#value 얻기
a={'car':['bus', 'truck', 'taxi'], 
    'train':'ktx'}
value=a.values()
print(value)
print(type(value))
print(value[0])
value2=list(value)
print(value2[0])

#요소 추가하기
a={'car':['bus', 'truck', 'taxi'], 
    'train':'ktx'}
a['plane']='jet'
print(a)
a['ship']=['boat', 'yacht']
print(a)

#요소 삭제하기
del a['car']
print(a)
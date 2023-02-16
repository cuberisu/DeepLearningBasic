# numpy의 array() 함수로 2차원 numpy배열 만들기
import numpy as np
my_arr = np.array([[10, 20, 30], [40, 50, 60]])
print(my_arr)
print(type(my_arr)) # type() 함수로 자료형 확인하기
print(my_arr[0][2]) # index로 요소 선택하기
print(np.sum(my_arr))   # numpy 내장함수 사용해보기: sum() 모든 요소 총합

# [[10 20 30]]
#  [40 50 60]]
# <class 'numpy.ndarray'>
# 30
# 210
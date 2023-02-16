# 문제 해결을 위해 당뇨병 환자의 데이터 준비하기
# 사이킷런은 다양한 데이터셋을 제공한다.

from sklearn.datasets import load_diabetes  # 당뇨병 관련 데이터를 이용한다.
diabetes = load_diabetes()
print(diabetes.data.shape, diabetes.target.shape)

print(type(diabetes))   # <class 'sklearn.utils._bunch.Bunch'>
print(type(diabetes.data))  # <class 'numpy.ndarray'>
print(type(diabetes.data.shape))    # <class 'tuple'>

# 사이킷런의 data 속성: 입력(x) / target 속성: 타깃(y)
# 두 속성 모두 numpy 배열
# numpy 배열에는 shape 속성이 있다.
# 이 shape 속성을 출력한다. (numpy 배열의 shape를 반환)

# (442, 10) (442,)
# tuple이다. tuple은 리스트와 유사하지만 수정 불가능
# tuple에 원소가 하나인 경우 반드시 ,를 끝에 써준다.
# 442개의 행, 10개의 열. 행은 샘플(데이터), 열은 특성이라고 한다.
# 선형회귀에서 특성 2개는 x, y축, 3개는 x, y, z 축 ...


# 입력 데이터와 타깃 데이터 자세히 보기
# 이 데이터의 의미가 무엇인지는 도메인 지식이 필요하므로 다루지 않는다.
print(diabetes.data[0:3])   # 0, 1, 2 인덱스의 행(데이터)만 출력. (슬라이싱)
print(diabetes.target[:3]) # 시작이 0이면 안 써줘도 된다.


# 데이터 시각화하기(이차원 그래프 그리기)
import matplotlib.pyplot as plt
plt.scatter(diabetes.data[:, 2], diabetes.target)
# [x축] :(처음부터 끝까지 모든 데이터(행)), 2(3번째 특성(열))
# [y축] target

plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 이후 코드를 간단히 쓰기 위해 x, y에 대입
x = diabetes.data[:, 2]
y = diabetes.target

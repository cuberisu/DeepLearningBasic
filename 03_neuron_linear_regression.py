# 선형 회귀를 위한 뉴런 만들기

# Neuron 클래스 만들기
""" 클래스: 변수, 메서드(클래스 안에 있는 함수) 등을 묶어놓은 것
파이썬에서는 관례 상 대문자로 시작
클래스들에 있는 메서드들은 첫 번째 매개변수는 self(객체 자기 자신)여야 함
객체 이름을 특정할 수 없으니 self라고 하는 것. """

from sklearn.datasets import load_diabetes  # 데이터셋
import matplotlib.pyplot as plt     # 그래프 그리기
diabetes = load_diabetes()  # 당뇨병 환자의 데이터 불러오기
x = diabetes.data[:, 2]     # 입력
y = diabetes.target         # 타깃

class Neuron:  # 객체를 만들 땐 n = Neuron() 형식으로 만든다.
    
    # 객체를 만들 때 맨 처음 실행되는 생성자 __init__()
    def __init__(self): # self에 객체 이름이 들어간다.
        self.w = 1.0    # w, b를 1.0으로 초기화
        self.b = 1.0

    # 정방향 계산 만들기 (forward propagation, 순전파?)
    # 예측값을 계산.
    def forpass(self, x): 
        y_hat = x * self.w + self.b
        return y_hat
    
    # 역방향 계산 만들기 (Backpropagation, 역전파)
    def backprop(self, x, err):
        w_grad = x * err
        b_grad = 1 * err
        return w_grad, b_grad
    
    # 훈련을 위한 fit() 메서드 구현
    def fit(self, x, y, epochs=100):
        for i in range(epochs):         # 에포크만큼 반복
            for x_i, y_i in zip(x, y):  # 모든 샘플에 대해 반복
                y_hat = self.forpass(x_i)   # 정방향 계산
                err = -(y_i - y_hat)        # 오차 계산
                w_grad, b_grad = self.backprop(x_i, err) # 역방향 계산
                self.w -= w_grad    # 가중치 업데이트
                self.b -= b_grad    # 절편 업데이트


n = Neuron()    # n 객체를 만들자. 
                # 객체를 만들자마자 init 메서드가 호출되며 self에 n이 대입된다.
                # 정의할 때와 달리 사용할 땐 self값은 넣어줄 필요가 없다.
                
print(n.w)  # 1.0
print(n.b)  # 1.0

n.fit(x, y) # 뉴런 훈련

print(n.w)
print(n.b)

# 그래프 그리기
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * n.w + n.b)
pt2 = (0.15, 0.15 * n.w + n.b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()    # 그래프 보이기
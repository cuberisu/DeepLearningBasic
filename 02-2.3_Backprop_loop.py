# for 중첩을 사용한 오차역전파 Backpropagation
# 이 모델로 y값을 예측도 해 보자.

from sklearn.datasets import load_diabetes
import matplotlib.pyplot as plt
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

# 초기 예측값
w = 1.0
b = 1.0
print("초기 가중치:", w)
print("초기 절편:", b)

# for문 중첩으로 더 많이 반복
for i in range(1, 100): # 100번 더 수행
    for x_i, y_i in zip(x, y):  # zip은 x와 y에 있는 원소들을 하나씩 꺼내서 tuple로 만들어줌
        y_hat = x_i * w + b
        err = y_i - y_hat
        w_rate = x_i
        w = w + w_rate * err
        b = b + 1 * err

print("예측 가중치:", w)
print("예측 절편:", b)

# 산점도 그리기
plt.scatter(x, y)
pt1 = (-0.1, -0.1 * w + b)  # -0.1 ~ 0.15 까지의 x값
pt2 = (0.15, 0.15 * w + b)
plt.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]])
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# 모델로 예측하기
x_new = 0.18
y_pred = x_new * w + b
print("x = 0.18 일 때, y =", y_pred)
plt.scatter(x, y)
plt.scatter(x_new, y_pred)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
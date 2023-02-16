# 오차역전파 Backpropagation
# 경사하강법의 한계점을 보완하는 방법
# 오차와 변화율을 곱하여 가중치와 절편을 업데이트

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

# 초기 예측값
w = 1.0
b = 1.0
y_hat = x[0] * w + b
print("초기 가중치:", w)
print("초기 절편:", b)

# 경사하강법 가중치 조절
w_inc = w + 0.1 # 가중치를 0.1 증가시켜서 예측값을 바꾼다.
y_hat_inc = x[0] * w_inc + b
w_rate = (y_hat_inc - y_hat) / (w_inc - w)  # 변화율. x[0]과 같다.
w_new = w + w_rate  # 변화율을 더해서 예측값을 바꾼다.(증가시킨다)
print("경사하강법 가중치:", w_new)

# 경사하강법 절편 조절
b_inc = b + 0.1 # 절편을 0.1 증가시켜서 예측값을 바꾼다.
y_hat_inc = x[0] * w + b_inc
b_rate = (y_hat_inc - y_hat) / (b_inc - b)  # 변화율. 1과 같다.
b_new = b + 1   # 변화율을 더해서 예측값을 바꾼다.(증가시킨다)
print("경사하강법 절편:", b_new)

# 오차와 변화율을 곱하자.
err = y[0] - y_hat
w_new = w + w_rate * err
b_new = b + 1 * err
print("오차역전파 가중치:", w_new)
print("오차역전파 절편:", b_new)

# 두 번째 샘플을 사용해서 w와 b를 계산하기
y_hat = x[1] * w_new + b_new
err = y[1] - y_hat
w_rate = x[1]
w_new = w_new + w_rate * err
b_new = b_new + b_rate * err
print("오차역전파2 가중치:", w_new)
print("오차역전파2 절편:", b_new)
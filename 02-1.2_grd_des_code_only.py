from sklearn.datasets import load_diabetes
diabetes = load_diabetes()
x = diabetes.data[:, 2]
y = diabetes.target

# 초기 예측값
w = 1.0
b = 1.0
y_hat = x[0] * w + b

# 가중치 조절
w_inc = w + 0.1 # 가중치를 0.1 증가시켜서 예측값을 바꾼다.
y_hat_inc = x[0] * w_inc + b

w_rate = (y_hat_inc - y_hat) / (w_inc - w)  # x[0]과 같다.
w_new = w + w_rate  # 변화율을 더해서 예측값을 바꾼다.(증가시킨다)
y_hat_w_new = x[0] * w_new + b

# 절편 조절
b_inc = b + 0.1 # 절편을 0.1 증가시켜서 예측값을 바꾼다.
y_hat_inc = x[0] * w + b_inc

b_rate = (y_hat_inc - y_hat) / (b_inc - b)  # 1과 같다.
b_new = b + 1   # 변화율을 더해서 예측값을 바꾼다.(증가시킨다)
y_hat_b_new = x[0] * w + b_new
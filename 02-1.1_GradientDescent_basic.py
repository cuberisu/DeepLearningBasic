# 경사하강법(Gradient Descent)
# 실제로 훈련 데이터에 맞는 w와 b 찾아보기

'''
이 방법의 문제점
y_hat이 y에 한참 미치지 못하는 값인 경우 w와 b를 더 큰 폭으로 수정할 수 없다.
y_hat이 y보다 커지면 y_hat을 감소시키기 못한다.
'''

from sklearn.datasets import load_diabetes  # 데이터 제공 라이브러리 사이킷런
# 당뇨병 관련 데이터를 이용한다.
diabetes = load_diabetes()  # class 'sklearn.utils._bunch.Bunch'

# 사이킷런의 data 속성: 입력(x) / target 속성: 타깃(y)
# 두 속성 모두 numpy 배열
# 이 데이터의 의미가 무엇인지는 도메인 지식이 필요하므로 다루지 않는다.

# 코드를 간단히 쓰기 위해 x, y에 대입
x = diabetes.data[:, 2]
y = diabetes.target

print("가중치(w)와 절편(b)으로 예측값(y_hat)을 만들고 실제 값(y)과 비교하자")
print("이 방법은 y_hat < y일 경우에만 적용할 수 있다.")
print("식은 y_hat = x[0] * w + b 이다.")


# 1. 무작위로 w와 b를 정한다. (무작위로 모델 만들기)
# 임의의 값으로 시작
w = 1.0
b = 1.0

print("처음 가중치:", w)
print("처음 절편:", b)


# 2. x에서 샘플 하나를 선택하여 y_hat을 계산한다. (무작위로 모델 예측하기)
# 첫 번째 샘플에 대한 예측 만들기
y_hat = x[0] * w + b
print("처음 예측값(y_hat):", y_hat)


# 3. y_hat과 선택한 샘플의 진짜 y값을 비교한다. (예측한 값과 진짜 정답 비교하기, 틀릴 확률 99%)
# 첫 번째 샘플의 실제 타깃
print("실제 값(y[0]):", y[0])


# 4. y_hat이 y와 더 가까워지도록 w, b를 조정한다. (모델 조정하기)
# w를 0.1만큼 증가시키기
w_inc = w + 0.1
y_hat_inc = x[0] * w_inc + b
print("w_inc = w + 0.1")
print("가중치로만 조정한 예측값(y_hat_inc):", y_hat_inc)


# 변화율
# y 변화량 / w 변화량
w_rate = (y_hat_inc - y_hat) / (w_inc - w)  # x[0]과 같다.
print("가중치 변화율(w_rate, x[0]):", w_rate, x[0])

# 변화율 부호에 따라 가중치를 업데이트
w_new = w + w_rate
y_hat_w_new = x[0] * w_new + b
print("새 가중치(w_new):", w_new)
print("새 예측 값:", y_hat_w_new)
# 변화율이 양수일 때: w가 증가하면 y_hat 증가
# 변화율이 음수일 때: w가 감소하면 y_hat 증가
# 즉 변화율을 더하면 예측값이 증가한다!


# 변화율로 절편 업데이트
b_inc = b + 0.1
y_hat_inc = x[0] * w + b_inc
print("b_inc = b + 0.1")
print("절편으로만 조정한 예측값(y_hat_inc):", y_hat_inc)

b_rate = (y_hat_inc - y_hat) / (b_inc - b)  # 1과 같다.
print("절편 변화율(b_rate):", b_rate)

b_new = b + 1
y_hat_b_new = x[0] * w + b_new
print("새 절편(b_new):", b_new)
print("새 예측 값:", y_hat_b_new)


# 5. 모든 샘플을 처리할 때까지 다시 2~4 항목을 반복한다.
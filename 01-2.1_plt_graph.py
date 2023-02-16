# matplotlib는 그래프 관련 표준 라이브러리
# show() 함수로 그래프를 볼 수 있다.

# plot() 함수로 그래프를 그려보자 
import matplotlib.pyplot as plt
plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])    # x축, y축의 값을 파이썬 리스트로 전달
plt.show()

# scatter() 함수로 산점도를 그려보자
plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) # x축, y축의 값을 파이썬 리스트로 전달
plt.show()
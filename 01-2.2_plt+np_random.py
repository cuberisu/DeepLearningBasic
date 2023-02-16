# numpy 배열을 이용한 랜덤 산점도 그리기
# numpy 배열로 난수를 만든다.
# matplotlib으로 그래프 그리고 보여준다.

import numpy as np
import matplotlib.pyplot as plt

# 정규분포란 가운데가 높고 양극단으로 갈수록 낮아져서 종 모양을 그리는 분포이다.
# random.randn(): 평균이 0이고 표준편차가 1인 정규분포에서 난수를 발생
x = np.random.randn(1000)   # 표준 정규 분포를 따르는 난수 1000개 생성
y = np.random.randn(1000)   # 표준 정규 분포를 따르는 난수 1000개 생성
plt.scatter(x, y)
plt.show()
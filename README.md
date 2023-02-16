TensorFlow is an open source software library for numerical computation using data flow graphs.  
TensorFlow는 Data flow graph를 사용하여 수치 연산을 하는 오픈소스 소프트웨어 라이브러리.  

유연한 아키텍처로 구성되어 있어 코드 수정없이 데스크탑, 서버 혹은 모바일 디바이스에서 CPU나 GPU를 사용하여 연산을 구동시킬 수 있다. TensorFlow는 원래 머신러닝과 딥 뉴럴 네트워크 연구를 목적으로 구글의 인공지능 연구 조직인 Google 브레인 팀의 연구자와 엔지니어들에 의해 개발되었다. 하지만 이 시스템은 여러 다른 분야에도 충분히 적용될 수 있다.  

Python을 활용하여 연산처리를 작성할 수 있다.   
다른 언어들도 대부분 지원하지만 Python 관련 자료가 가장 많다.  

TensorFlow는 graph로 computation하는 프로그래밍 시스템.  
연산은 graph로 표현, 데이터는 tensor로 표현한다.  
graph에 있는 node는 op(작업, operation)라고 부른다.  
node는 수치 연산을 나타내고,  
edge는 node 사이를 이동하는 tensor(다차원 데이터 배열, a multidimensional array of elements)를 나타낸다.  

[Tensor Ranks, Shapes, Type(GitBook)](https://tensorflowkorea.gitbooks.io/tensorflow-kr/content/g3doc/resources/dims_types.html)  

Installing TensorFlow  
Windows: `pip install tensorflow`  

설치 확인  
`pip show tensorflow`  

Install TF version 1  
`pip install tensorflow==1.2.1`  
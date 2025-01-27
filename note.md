# installation (venv)
pip install Flask 
pip install pandas numpy scikit-learn dash plotly
py -m pip install markdown

# Dash
https://positive-impactor.tistory.com/1081




# 과제 개요
과제 개요
스마트 농업에서는 환경 데이터를 활용하여 토양의 습도를 모니터링하고 예측할 수 있습니다.
주어진 데이터를 기반으로 토양 습도 수준이 높거나 낮은지를 예측하는 로지스틱 회귀 모델을 구현하고,
이를 대시보드로 시각화하세요.
============================================================
# Google Colab 환경 설정
!pip install dash pandas sklearn plotly
!pip install dash
!pip install jupyter-dash
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


문제 1: 데이터셋 생성 및 탐색
1. 아래 주어진 데이터를 이용하여 pandas과 numpy 를 사용해 데이터프레임을 생성하세요.
 - 아래의 데이터 갯수는 각각 1000개를 만든다.
 - 기온 (10~35°C 범위) => Temperature
 - 습도 (20~90% 범위) => Humidity
 - 토양 유형 (Sandy, Clay, Loamy) =>토양변수 SoilType , 유형 ['Sandy', 'Clay', 'Loamy']
 - 강수량 (0~200 mm 범위) => Rainfall
 - 풍속 (0~15 m/s 범위) => WindSpeed
 - 토양 습도 (10~50% 범위) => SoilMoisture
2. 생성된 데이터셋의 처음 5개 행을 출력하세요.
3. 데이터의 기본 통계를 확인하고 ('.describe()' 활용), 결측값 여부를 점검하세요.
4. 'Temperature', 'Humidity', 'Rainfall'의 분포를 히스토그램으로 시각화하세요.
**힌트:** pd.DataFrame() , df.describe() , seaborn.histplot()을 사용하세요.


============================================================
문제 2: 데이터 전처리
1. 토양 유형(SoilType)과 같은 범주형 변수를 원-핫 인코딩하여 숫자로 변환하세요.

2. SoilMoisture 변수를 30% 기준으로 이진 분류(0: 낮음, 1: 높음) 하세요.
3. 데이터를 훈련 세트(80%)와 테스트 세트(20%)로 분할하세요.
4. 데이터 표준화를 수행하고 평균과 표준 편차를 출력하세요.
**힌트:** pd.get_dummies(), train_test_split(), StandardScaler() 활용


============================================================
문제 3: 로지스틱 회귀 모델 구축
1. LogisticRegression을 사용하여 모델을 학습하세요. (랜덤 시드: 42, 최대 반복 횟수: 1000)
2. 테스트 데이터를 이용해 예측을 수행하세요.
3. 모델의 정확도(accuracy)를 출력하세요.
**힌트:** fit(), predict(), accuracy_score() 활용
============================================================
문제 4: 모델 평가 및 성능 분석
1. 혼동 행렬(Confusion Matrix)을 생성하고 시각화하세요.
![Image](https://github.com/user-attachments/assets/2ae261cf-a753-433b-886e-44fc497578b7)


2. 정밀도(Precision), 재현율(Recall), F1-score를 계산하세요.
3. 실제와 예측된 값을 비교하는 산점도를 생성하세요.
힌트: confusion_matrix(), classification_report(), plotly.express.imshow()
============================================================
문제 5: 대시보드 구현
Dash를 이용하여 다음 기능이 포함된 대시보드를 구축하세요.
1. 데이터 시각화 섹션
 - Temperature, Humidity의 분포를 히스토그램으로 표시.
2. 모델 평가 섹션
 - 정확도, 혼동 행렬을 시각화.
3. 사용자 입력 기반 예측 섹션
 - 사용자가 Temperature, Humidity, Rainfall, WindSpeed, SoilType 값을 조정하면 예측 결과를 출력.
**힌트:**
- dash.Dash(), dcc.Graph(), html.Div(), app.run_server(debug=True) 활용
- predict()를 이용해 사용자 입력값을 바탕으로 토양 습도 예측


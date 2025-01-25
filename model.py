import pandas as pd
import numpy as np
from dash import callback
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from dash.dependencies import Input, Output
import plotly.express as px
import config

model = None
X_train = None
X_test = None
y_train = None
y_test = None


def train_model():
    global model, X_train, X_test, y_train, y_test

    weather_dict = {
        'Temperature': np.random.uniform(10, 35, config.MAX_SIZE).tolist(),
        'Humidity': np.random.uniform(20, 90, config.MAX_SIZE).tolist(),
        'SoilType': np.random.choice(config.SOIL_TYPES, size=config.MAX_SIZE).tolist(),
        'Rainfall': np.random.uniform(0, 200, config.MAX_SIZE).tolist(),
        'WindSpeed': np.random.uniform(0, 15, config.MAX_SIZE).tolist(),
        'SoilMoisture': np.random.uniform(10, 50, config.MAX_SIZE).tolist()
    }
    df = pd.DataFrame(weather_dict)

    # 이진 분류 생성
    df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > config.TARGET_THRESHOLD, 1, 0)

    # 특성과 타겟 분리
    X = df.drop(columns=['SoilMoisture', 'SoilMoistureDegree'])
    y = df['SoilMoistureDegree']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ### [원-핫 인코딩] OneHotEncoder 방식 ###
    # 원-핫 인코딩 및 표준화 스케일링을 포함하는 파이프라인 정의
    # OneHotEncoder를 사용하여 훈련 세트와 테스트 세트에 대해 동일한 인코딩을 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Temperature', 'Humidity', 'Rainfall', 'WindSpeed']),
            ('cat', OneHotEncoder(drop='first'), ['SoilType'])
        ]
    )

    # 모델 파이프라인 정의
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 모델 학습
    model.fit(X_train, y_train)
    return df


# 모델 평가 결과 업데이트
@callback(
    Output('model-evaluation', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_model_evaluation(n_clicks):
    global model, X_train, X_test, y_train, y_test

    print("model-evaluation is called...")

    # 정확도 계산
    accuracy = accuracy_score(y_test, model.predict(X_test))
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))

    # 혼동 행렬 시각화
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Low', 'High'],
                    y=['Low', 'High'],
                    color_continuous_scale='Blues',
                    title='혼동 행렬')

    return f"모델 정확도: {accuracy:.2f}", fig


# 사용자 입력 기반 예측
@callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('temp-input', 'value'),
    Input('hum-input', 'value'),
    Input('rain-input', 'value'),
    Input('wind-input', 'value'),
    Input('soil-type-input', 'value')
)
def predict_soil_moisture(n_clicks, temp, hum, rain, wind, soil_type):
    global model  # X_train, X_test, y_train, y_test

    print("predict_soil_moisture is called...")

    if n_clicks is None:
        return ""

    # 기본값이 유효한지 확인
    if temp is None or hum is None or rain is None or wind is None or soil_type is None:
        return "모든 입력값을 입력해 주세요."

    # 사용자 입력을 DataFrame으로 변환
    input_data = pd.DataFrame([[temp, hum, rain, wind, soil_type]],
                              columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed', 'SoilType'])

    # 예측 수행
    prediction = model.predict(input_data)[0]
    # print(model.predict(input_data))

    # return f"예측된 SoilMoistureDegree: {'높음' if prediction == 1 else '낮음'}"
    return f"토양습도 : {'높음' if prediction == 1 else '낮음'}"

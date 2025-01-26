import config
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors

import dash
from dash import callback
from dash.dependencies import Input, Output

import plotly.express as px

df = None
model = None
X_train = None
X_test = None
y_train = None
y_test = None


def train_model():
    global df, model, X_train, X_test, y_train, y_test
    df = pd.DataFrame(config.weather_dict)

    """이진 분류 생성"""
    df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > config.TARGET_THRESHOLD, 1, 0)

    """특성과 타겟 분리"""
    X = df.drop(columns=['SoilMoisture', 'SoilMoistureDegree'])
    y = df['SoilMoistureDegree']

    """데이터 분할"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    """원-핫 인코딩 및 표준화 스케일링을 포함하는 파이프라인 정의"""
    # OneHotEncoder를 사용하여 훈련 세트와 테스트 세트에 대해 동일한 인코딩을 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Temperature', 'Humidity', 'Rainfall', 'WindSpeed']),
            ('cat', OneHotEncoder(drop='first'), ['SoilType'])
        ]
    )

    """모델 파이프라인 정의 (로지스틱회귀)"""
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    """모델 학습"""
    model.fit(X_train, y_train)

    """학습된 모델 출력"""
    # 1. 전처리기(Preprocessor) 출력
    preprocessor = model.named_steps['preprocessor']

    # 2. 수치형 변수의 평균과 표준편차 출력
    if hasattr(preprocessor.named_transformers_['num'], 'mean_'):
        print("수치형 변수 평균:", preprocessor.named_transformers_['num'].mean_)
        print("수치형 변수 표준편차:", preprocessor.named_transformers_['num'].scale_)

    # 3. 원-핫 인코더의 변환된 클래스 출력
    if hasattr(preprocessor.named_transformers_['cat'], 'categories_'):
        print("원-핫 인코딩된 클래스:", preprocessor.named_transformers_['cat'].categories_)

    # 4. 모델의 계수 출력 (Logistic Regression 예시)
    model_coefficients = model.named_steps['classifier'].coef_
    model_intercept = model.named_steps['classifier'].intercept_

    print("모델 계수:", model_coefficients)
    print("모델 절편:", model_intercept)

    return df


@callback(
    Output('model-evaluation', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('predict-button', 'n_clicks')
)
def show_model_evaluation_and_confusion_matrix(n_clicks):  # def update_model_evaluation(n_clicks):
    """모델 평가 결과 업데이트 함수"""
    """
    - 프로그램로드시 모델학습되고 모델정확도 출력 및 혼돈매트릭스가 화면에 그려진다. 
    - 프로그램로드이후, 사용자 입력이 (현재로서는) 기존의 로직에 전혀 영향을 주지 않으므로
      n_clicks시 기존 Output을 리턴
    """

    global model, X_train, X_test, y_train, y_test

    if n_clicks:
        return dash.no_update, dash.no_update

    print("model-evaluation is called...")

    # 정확도 계산
    accuracy = accuracy_score(y_test, model.predict(X_test))
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))

    print("model: ", model)
    print("accuracy: ", accuracy)
    print("conf_matrix: ", conf_matrix)

    # 혼동 행렬 시각화
    fig = px.imshow(conf_matrix,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Low', 'High'],
                    y=['Low', 'High'],
                    color_continuous_scale=px.colors.sequential.Magenta,
                    title='Confusion Matrix')

    return f"Accuracy: {accuracy:.2f}", fig


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
    """사용자 입력 기반하여 토양 습도 예측하는 함수"""

    global df, model  # X_train, X_test, y_train, y_test
    max_neighbors = 3

    print("predict_soil_moisture is called...")

    if n_clicks is None:
        return ""

    # 기본입력값이 유효한지 확인
    if temp is None or hum is None or rain is None or wind is None or soil_type is None:
        return "모든 입력값을 입력해 주세요."

    # 사용자 입력을 DataFrame으로 변환
    input_data = pd.DataFrame([[temp, hum, rain, wind, soil_type]],
                              columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed', 'SoilType'])

    # SoilMoistureDegree 예측
    prediction = model.predict(input_data)[0]

    # 최근접 이웃 검색을 위한 데이터 준비
    numeric_input = input_data[['Temperature', 'Humidity', 'Rainfall', 'WindSpeed']].values
    numeric_df = df[['Temperature', 'Humidity', 'Rainfall', 'WindSpeed']].values

    # SoilMoistureDegree 값 추가
    # df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > config.TARGET_THRESHOLD, 1, 0)

    # 최근접 이웃 모델 생성
    nbrs = NearestNeighbors(n_neighbors=max_neighbors)
    nbrs.fit(numeric_df)

    # 가장 가까운 이웃 찾기
    distances, indices = nbrs.kneighbors(numeric_input)

    # 가장 가까운 이웃의 SoilMoisture 값과 SoilMoistureDegree 가져오기
    closest_soil_moisture = df.iloc[indices[0][0]]['SoilMoisture']
    closest_soil_moisture_degree = df.iloc[indices[0][0]]['SoilMoistureDegree']


    """토양 습도 상태(높음, 낮음) 및 트양습고 값 처리"""
    """
    SoilMoistureDegree는 30을 기준으로 크면 1 작으면 0으로 설정되어 있다.
    하지만 현재 최근접 이웃 3개의 SoilMoisture값이 30이상 일 수도 이하일 수도 있다.
    즉, 현재로서는 예측된 SoilMoistureDegree (prediction)과 최근접 이웃의 soil_moisture_degree 가 같으면 (1 또는 0)
    최근접 이웃의 SoilMoisture 값을 리턴하는 것이 최선
    """
    result_text = ''
    if prediction == closest_soil_moisture_degree:
        result_text = f"토양습도: {closest_soil_moisture:.2f}, " + "\n" \
                      f"토양습도정도: {'높음' if prediction == 1 else '낮음'}, " + "\n" \
                      f"최근접 {max_neighbors}개 이웃의 토양습도정도: {'높음' if closest_soil_moisture_degree == 1 else '낮음'}"
        return result_text
    return "예측된 토양습도값이 없습니다. 위의 입력값들을 다시 설정해 보세요!"




        # result_text = "예측이 일치합니다."
        # result_text = f"예측된 SoilMoistureDegree: {'높음' if prediction == 1 else '낮음'}, \
        #               실제 SoilMoisture: {closest_soil_moisture:.2f}, \
        #               이웃의 SoilMoistureDegree: {'높음' if closest_soil_moisture_degree == 1 else '낮음'}"
        # return result_text

# def predict_soil_moisture(n_clicks, temp, hum, rain, wind, soil_type):
#     global df, model  # X_train, X_test, y_train, y_test
#
#     print("predict_soil_moisture is called...")
#
#     if n_clicks is None:
#         return ""
#
#     # 기본값이 유효한지 확인
#     if temp is None or hum is None or rain is None or wind is None or soil_type is None:
#         return "모든 입력값을 입력해 주세요."
#
#     # 사용자 입력을 DataFrame으로 변환
#     input_data = pd.DataFrame([[temp, hum, rain, wind, soil_type]],
#                               columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed', 'SoilType'])
#
#     # 예측 수행
#     prediction = model.predict(input_data)[0]  # soil_moisture_prediction
#     degree = 1 if prediction > config.TARGET_THRESHOLD else 0
#
#     return f"토양습도: {prediction:.2f}, ({'높음' if degree == 1 else '낮음'})"

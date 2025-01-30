import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# 데이터 생성
MAX_SIZE = 1000
soil_types = ['Sandy', 'Clay', 'Loamy']

weather_dict = {
    'Temperature': np.random.uniform(10, 35, MAX_SIZE).tolist(),
    'Humidity': np.random.uniform(20, 90, MAX_SIZE).tolist(),
    'SoilType': np.random.choice(soil_types, size=MAX_SIZE).tolist(),
    'Rainfall': np.random.uniform(0, 200, MAX_SIZE).tolist(),
    'WindSpeed': np.random.uniform(0, 15, MAX_SIZE).tolist(),
    'SoilMoisture': np.random.uniform(10, 50, MAX_SIZE).tolist()
}

df = pd.DataFrame(weather_dict)

# 이진 분류 생성
TARGET_THRESHOLD = 30
df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > TARGET_THRESHOLD, 1, 0)

# 특성과 타겟 분리
X = df.drop(columns=['SoilMoisture', 'SoilMoistureDegree'])
y = df['SoilMoistureDegree']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#######################################################################
# pd.get_dummies() 사용시, 훈련 데이터셋(train)과 테스트 데이터셋(test) 간의
# 특성(컬럼) 이름이 일치하지 않을 때 발생.
# df = pd.get_dummies(df, columns=['SoilType'], drop_first=True)
#######################################################################

#######################################################################
# ERROR EXISTS [원-핫 인코딩] pd.get_dummies() 방식
# 훈련 세트와 테스트 세트를 합쳐서 원-핫 인코딩
# X_combined = pd.concat([X_train, X_test])
# X_combined = pd.get_dummies(X_combined, columns=['SoilType'], drop_first=True)
#
# # 다시 훈련 세트와 테스트 세트로 분할
# X_train = X_combined.iloc[:len(X_train)]
# X_test = X_combined.iloc[len(X_train):]
#
# # 데이터 표준화
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)  # fit_transform 사용
# X_test_scaled = scaler.transform(X_test)
#
# # 모델 학습
# model = LogisticRegression(max_iter=1000, random_state=42)
# model.fit(X_train_scaled, y_train)

#######################################################################
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


# 대시보드 앱 생성
app = dash.Dash(__name__)

# 레이아웃 정의
app.layout = html.Div([
    html.H1("스마트 농업 토양 습도 예측 대시보드"),

    html.Div([
        html.H2("데이터 시각화 섹션"),
        dcc.Graph(
            id='histogram-temp',
            figure=px.histogram(
                df,
                x='Temperature',
                nbins=30,
                title='Temperature 분포',
                color_discrete_sequence=px.colors.qualitative.Prism,  # 색상 변경
                opacity=0.7,  # 투명도 조정
                histnorm='percent',  # 히스토그램 정규화 (필요에 따라 조정)
                # barmode='overlay'  # 바 모드 설정 (필요에 따라 조정)
            )
        ),
        dcc.Graph(
            id='histogram-hum',
            figure=px.histogram(
                df,
                x='Humidity',
                nbins=30,
                title='Humidity 분포',
                color_discrete_sequence=px.colors.qualitative.Plotly,  # 색상 변경
                opacity=0.7,  # 투명도 조정
                histnorm='percent',  # 히스토그램 정규화 (필요에 따라 조정)
                # barmode='overlay'  # 바 모드 설정 (필요에 따라 조정)
            )
        ),
    ]),

    html.Div([
        html.H2("모델 평가 섹션"),
        html.Div(id='model-evaluation'),
        dcc.Graph(id='confusion-matrix'),
    ]),

    html.Label("Temperature (°C):", className='label'),
    dcc.Slider(
        id='temp-input',
        min=10,
        max=33,
        value=20,  # 기본값
        marks={i: str(i) for i in range(10, 36)},  # 슬라이더 마크 설정
        step=1,
        className='input'
    ),

    html.Label("Humidity (%):", className='label'),
    dcc.Slider(
        id='hum-input',
        min=20,
        max=100,
        value=50,  # 기본값
        marks={i: str(i) for i in range(20, 91, 10)},  # 슬라이더 마크 설정
        step=1,
        className='input'
    ),

    html.Label("Rainfall (mm):", className='label'),
    dcc.Slider(
        id='rain-input',
        min=0,
        max=201,
        value=120,  # 기본값
        marks={i: str(i) for i in range(0, 201, 20)},  # 슬라이더 마크 설정
        step=1,
        className='input'
    ),

    html.Label("Wind Speed (m/s):", className='label'),
    dcc.Slider(
        id='wind-input',
        min=0,
        max=16,
        value=5,  # 기본값
        marks={i: str(i) for i in range(0, 16)},  # 슬라이더 마크 설정
        step=1,
        className='input'
    ),

    # html.Label("Soil Type:", className='label'),
    # dcc.Dropdown(
    #     id='soil-type-input',
    #     options=[{'label': soil, 'value': soil} for soil in soil_types],
    #     value='Sandy',  # 기본값
    #     className='input'
    # ),

    html.Label("Soil Type:", className='label'),
    dcc.RadioItems(
        id='soil-type-input',
        options=[{'label': soil, 'value': soil} for soil in soil_types],
        value='Sandy',  # 기본값
        labelStyle={'display': 'block'},  # 각 옵션을 블록으로 표시
        className='input'
    ),
    html.Button('예측하기', id='predict-button', className='button'),
    html.Div(id='prediction-output', className='output')

    # html.Div([
    #     html.H2("사용자 입력 기반 예측 섹션"),
    #     html.Label("Temperature:"),
    #     dcc.Input(id='temp-input', type='number', value=25),  # 기본값 25
    #     html.Label("Humidity:"),
    #     dcc.Input(id='hum-input', type='number', value=50),  # 기본값 50
    #     html.Label("Rainfall:"),
    #     dcc.Input(id='rain-input', type='number', value=100),  # 기본값 100
    #     html.Label("WindSpeed:"),
    #     dcc.Input(id='wind-input', type='number', value=5),  # 기본값 5
    #     html.Label("SoilType:"),
    #     dcc.Dropdown(
    #         id='soil-type-input',
    #         options=[{'label': soil, 'value': soil} for soil in soil_types],
    #         value='Sandy'  # 기본값
    #     ),
    #     html.Button('예측하기', id='predict-button'),
    #     html.Div(id='prediction-output')
    # ])
])


# 모델 평가 결과 업데이트
@app.callback(
    Output('model-evaluation', 'children'),
    Output('confusion-matrix', 'figure'),
    Input('predict-button', 'n_clicks')
)
def update_model_evaluation(n_clicks):
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
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    Input('temp-input', 'value'),
    Input('hum-input', 'value'),
    Input('rain-input', 'value'),
    Input('wind-input', 'value'),
    Input('soil-type-input', 'value')
)
def predict_soil_moisture(n_clicks, temp, hum, rain, wind, soil_type):
    if n_clicks is None:
        return "예측 결과는 여기에 표시됩니다."

    # 기본값이 유효한지 확인
    if temp is None or hum is None or rain is None or wind is None:
        return "모든 입력값을 입력해 주세요."

    # 사용자 입력을 DataFrame으로 변환
    input_data = pd.DataFrame([[temp, hum, rain, wind, soil_type]],
                              columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed', 'SoilType'])

    # 예측 수행
    prediction = model.predict(input_data)[0]

    print(model.predict(input_data))

    # return f"예측된 SoilMoistureDegree: {'높음' if prediction == 1 else '낮음'}"
    return f"토양습도 : {'높음' if prediction == 1 else '낮음'}"


# 서버 실행
if __name__ == '__main__':
    app.run_server(debug=True)

########################
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix, accuracy_score
# import dash
# from dash import dcc, html
# from dash.dependencies import Input, Output
# import plotly.express as px
#
# # 데이터 생성
# MAX_SIZE = 1000
# TARGET_THRESHOLD = 30
# soil_types = ['Sandy', 'Clay', 'Loamy']
#
# weather_dict = {
#     'Temperature': np.random.uniform(10, 35, MAX_SIZE).tolist(),
#     'Humidity': np.random.uniform(20, 90, MAX_SIZE).tolist(),
#     'SoilType': np.random.choice(soil_types, size=MAX_SIZE).tolist(),
#     'Rainfall': np.random.uniform(0, 200, MAX_SIZE).tolist(),
#     'WindSpeed': np.random.uniform(0, 15, MAX_SIZE).tolist(),
#     'SoilMoisture': np.random.uniform(10, 50, MAX_SIZE).tolist()
# }
#
# df = pd.DataFrame(weather_dict)
#
# # 이진 분류 생성
# # TARGET_THRESHOLD = 30
# df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > TARGET_THRESHOLD, 1, 0)
#
# # 데이터 준비
# df = pd.get_dummies(df, columns=['SoilType'], drop_first=True)  # 원-핫 인코딩
# data = df.drop(columns=['SoilMoisture', 'SoilMoistureDegree'])
# target = df['SoilMoistureDegree']
#
# # 데이터 분할
# train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.2, random_state=42)
#
# # 데이터 표준화
# scaler = StandardScaler()
# train_scaled = scaler.fit_transform(train_input)  # fit_transform 사용
# test_scaled = scaler.transform(test_input)
#
# # 모델 학습
# lr_model = LogisticRegression(max_iter=1000, random_state=42)
# lr_model.fit(train_scaled, train_target)
#
# # 대시보드 앱 생성
# app = dash.Dash(__name__)
#
# # 레이아웃 정의
# app.layout = html.Div([
#     html.H1("스마트 농업 토양 습도 예측 대시보드"),
#
#     html.Div([
#         html.H2("데이터 시각화 섹션"),
#         dcc.Graph(id='histogram-temp', figure=px.histogram(df, x='Temperature', nbins=30, title='Temperature 분포')),
#         dcc.Graph(id='histogram-hum', figure=px.histogram(df, x='Humidity', nbins=30, title='Humidity 분포')),
#     ]),
#
#     html.Div([
#         html.H2("모델 평가 섹션"),
#         html.Div(id='model-evaluation'),
#         dcc.Graph(id='confusion-matrix'),
#     ]),
#
#     html.Div([
#         html.H2("사용자 입력 기반 예측 섹션"),
#         html.Label("Temperature:"),
#         dcc.Input(id='temp-input', type='number', value=25),  # 기본값 25
#         html.Label("Humidity:"),
#         dcc.Input(id='hum-input', type='number', value=50),  # 기본값 50
#         html.Label("Rainfall:"),
#         dcc.Input(id='rain-input', type='number', value=100),  # 기본값 100
#         html.Label("WindSpeed:"),
#         dcc.Input(id='wind-input', type='number', value=5),  # 기본값 5
#         html.Label("SoilType:"),
#         dcc.Dropdown(
#             id='soil-type-input',
#             options=[{'label': soil, 'value': soil} for soil in soil_types],
#             value='Sandy'  # 기본값
#         ),
#         html.Button('예측하기', id='predict-button'),
#         html.Div(id='prediction-output')
#     ])
# ])
#
#
# # 모델 평가 결과 업데이트
# @app.callback(
#     Output('model-evaluation', 'children'),
#     Output('confusion-matrix', 'figure'),
#     Input('predict-button', 'n_clicks')
# )
# def update_model_evaluation(n_clicks):
#     # 정확도 계산
#     accuracy = accuracy_score(test_target, lr_model.predict(test_scaled))
#     conf_matrix = confusion_matrix(test_target, lr_model.predict(test_scaled))
#
#     # 혼동 행렬 시각화
#     fig = px.imshow(conf_matrix,
#                     labels=dict(x="Predicted", y="Actual", color="Count"),
#                     x=['Low', 'High'],
#                     y=['Low', 'High'],
#                     color_continuous_scale='Blues',
#                     title='혼동 행렬')
#
#     return f"모델 정확도: {accuracy:.2f}", fig
#
#
# # 사용자 입력 기반 예측
# @app.callback(
#     Output('prediction-output', 'children'),
#     Input('predict-button', 'n_clicks'),
#     Input('temp-input', 'value'),
#     Input('hum-input', 'value'),
#     Input('rain-input', 'value'),
#     Input('wind-input', 'value'),
#     Input('soil-type-input', 'value')
# )
# def predict_soil_moisture(n_clicks, temp, hum, rain, wind, soil_type):
#
#     if n_clicks is None:
#         return "예측 결과는 여기에 표시됩니다."
#
#     # 기본값이 유효한지 확인
#     if temp is None or hum is None or rain is None or wind is None:
#         return "모든 입력값을 입력해 주세요."
#
#     # 사용자 입력값 기반으로 예측 수행
#     # SoilType을 원-핫 인코딩된 형식으로 변환
#     soil_type_encoded = {
#         'Sandy': [1, 0, 0],  # SoilType_Sandy
#         'Clay': [0, 1, 0],  # SoilType_Clay
#         'Loamy': [0, 0, 1]  # SoilType_Loamy
#     }
#
#     print(soil_type)
#     print(soil_type_encoded[soil_type])
#
#     # 사용자 입력을 DataFrame으로 변환
#     input_data = pd.DataFrame([[temp, hum, rain, wind] + soil_type_encoded[soil_type]],
#                               columns=['Temperature', 'Humidity', 'Rainfall', 'WindSpeed', 'SoilType_Sandy',
#                                        'SoilType_Clay', 'SoilType_Loamy'])
#
#     print("input_data: ", input_data)
#
#     input_scaled = scaler.transform(input_data)  # DataFrame으로 변환 후 변환
#     print("input_scaled: ", input_scaled)
#
#     prediction = lr_model.predict(input_scaled)[0]
#     print("prediction: ", prediction)
#
#     return f"예측된 SoilMoistureDegree: {'높음' if prediction == 1 else '낮음'}"
#
#
# # 서버 실행
# if __name__ == '__main__':
#     app.run_server(debug=True)



# from flask import Flask, render_template
#
# app = Flask(__name__)
#
#
# @app.route('/')
# def index():
#     return render_template('index.html', msg='sdfdsfsda')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)

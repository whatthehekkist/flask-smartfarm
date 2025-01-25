import config
from dash import dcc, html
import plotly.express as px


def create_layout(df):
    # 레이아웃 정의
    return html.Div([
        html.H1("스마트 농업 토양 습도 예측 대시보드"),

        html.Div([
            html.H2("Temperature & Humidity Histogram"),
            dcc.Graph(
                id='histogram-temp',
                figure=px.histogram(
                    df,
                    x='Temperature',
                    nbins=30,
                    title='Temperature 분포',
                    color_discrete_sequence=px.colors.qualitative.G10,
                    opacity=0.7,
                    histnorm='percent',
                )
            ),
            dcc.Graph(
                id='histogram-hum',
                figure=px.histogram(
                    df,
                    x='Humidity',
                    nbins=30,
                    title='Humidity 분포',
                    color_discrete_sequence=px.colors.qualitative.Plotly,
                    opacity=0.7,  # 투명도 조정
                    histnorm='percent',
                )
            ),
        ]),

        html.Div([
            html.H2("Model Evaluation"),
            html.Div(id='model-evaluation'),
            dcc.Graph(id='confusion-matrix'),
        ]),

        html.Div([
            html.H2("Predict Soil Moisture", style={'textAlign': 'center'}),
            html.Label("Temperature (°C):", className='label'),
            dcc.Slider(
                id='temp-input',
                min=10,
                max=33,
                value=20,  # 기본값
                marks={i: str(i) for i in range(10, 36)},
                step=1,
                className='input',
            ),

            html.Label("Humidity (%):", className='label'),
            dcc.Slider(
                id='hum-input',
                min=20,
                max=100,
                value=50,  # 기본값
                marks={i: str(i) for i in range(20, 91, 10)},
                step=1,
                className='input',
            ),

            html.Label("Rainfall (mm):", className='label'),
            dcc.Slider(
                id='rain-input',
                min=0,
                max=201,
                value=120,  # 기본값
                marks={i: str(i) for i in range(0, 201, 20)},
                step=1,
                className='input'
            ),

            html.Label("Wind Speed (m/s):", className='label'),
            dcc.Slider(
                id='wind-input',
                min=0,
                max=16,
                value=5,  # 기본값
                marks={i: str(i) for i in range(0, 16)},
                step=1,
                className='input'
            ),

            html.Label("Soil Type:", className='label'),
            dcc.Dropdown(
                id='soil-type-input',
                options=[{'label': soil, 'value': soil} for soil in config.SOIL_TYPES],
                value='Sandy',  # 기본값
                className='input'
            ),

            # html.Label("Soil Type:", className='label'),
            # dcc.RadioItems(
            #     id='soil-type-input',
            #     options=[{'label': soil, 'value': soil} for soil in soil_types],
            #     value='Sandy',  # 기본값
            #     labelStyle={'display': 'block'},  # 각 옵션을 블록으로 표시
            #     className='input'
            # ),
            html.Button('예측하기', id='predict-button', className='button'),
            html.Div(id='prediction-output', className='output')
        ]),
    ])

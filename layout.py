import config
from dash import dcc, html
import plotly.express as px


def create_histogram_graph(_id, df, x_col, title):
    """히스토그램 그래프 생성 함수"""
    # color_sequence = px.colors.qualitative.Set1

    print(df.head())

    return dcc.Graph(
        id=_id,
        figure=px.histogram(
            df,
            x=x_col,
            nbins=30,
            title=title,
            # color=x_col,
            # color_discrete_sequence=px.colors.sequential.Viridis,
            # color_discrete_sequence=px.colors.sequential.Jet,
            # color_discrete_sequence=px.colors.qualitative.Vivid_r if x_col == 'Temperature' else px.colors.qualitative.Plotly,
            color_discrete_sequence=px.colors.sequential.Emrld if x_col == 'Temperature' else px.colors.sequential.Viridis,
            opacity=0.7,
            histnorm='percent',
        )
    )


def create_slider(_id, label, min_value, max_value, value, marks):
    """슬라이더 생성 함수"""
    return html.Div([
        html.Label(label, className='label'),
        dcc.Slider(
            id=_id,
            min=min_value,
            max=max_value,
            value=value,
            marks=marks,
            step=1,
            className='input'
        )
    ])


def create_dropdown(_id, label, options, value):
    """드롭다운 생성 함수"""
    return html.Div([
        html.Label(label, className='label'),
        dcc.Dropdown(
            id=_id,
            options=options,
            value=value,
            className='input'
        )
    ])


def create_layout(df):
    # 레이아웃 정의
    return html.Div([

        html.Div([
            html.H1("스마트 농업 토양 습도 예측 대시보드"),

            html.Div([
                html.H2("Temperature & Humidity Histogram"),
                create_histogram_graph('histogram-temp', df, 'Temperature', 'Temperature 분포'),
                create_histogram_graph('histogram-hum', df, 'Humidity', 'Humidity 분포'),
            ]),

            html.Div([
                html.H2("Model Evaluation"),
                html.Div(id='model-evaluation'),
                dcc.Graph(id='confusion-matrix'),
            ]),
        ], style={'textAlign': 'center'}),

        html.Div([
            html.H2("Predict Soil Moisture", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    create_slider('temp-input', "Temperature (°C):", 10, 36, 20, {i: str(i) for i in range(10, 37, 2)}),
                    create_slider('hum-input', "Humidity (%):", 20, 90, 50, {i: str(i) for i in range(20, 91, 10)}),
                    create_slider('rain-input', "Rainfall (mm):", 0, 200, 120, {i: str(i) for i in range(0, 201, 20)}),
                    create_slider('wind-input', "Wind Speed (m/s):", 0, 15, 5, {i: str(i) for i in range(0, 17)}),
                ], style={'paddingRight': '5%'}),
                create_dropdown('soil-type-input', "Soil Type:",
                                [{'label': soil, 'value': soil} for soil in config.SOIL_TYPES], 'Sandy'),
                html.Button('예측하기', id='predict-button', className='button', style={
                    'background': 'radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(70,252,216,1) 100%)'}),
            ], style={'width': '80%', 'margin': '0 auto'}),
            html.Div(html.H3(id='prediction-output', className='output', style={'padding-bottom': '5%'}))
        ]),
    ])

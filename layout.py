import config
from dash import dcc, html
from dash.dash_table import DataTable
import plotly.express as px


def create_histogram_graph(_id, df, x_col, title):
    """히스토그램 그래프 생성 함수"""
    # color_sequence = px.colors.qualitative.Set1

    fig = px.histogram(
        df,
        x=x_col,
        nbins=30,
        title=title,
        color_discrete_sequence=px.colors.sequential.Emrld if x_col == 'Temperature' else px.colors.sequential.Viridis,
        opacity=0.7,
        histnorm='percent',
    )
    fig.update_layout(
        # plot_bgcolor='lightblue',
        paper_bgcolor='rgb(248, 249, 250)',
        xaxis_title=x_col,
        yaxis_title='percentage'
    )
    return dcc.Graph(
        id=_id,
        figure=fig
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


def show_weather_data_table(df):
    """날씨 데이터 테이블로 보여주는 함수"""

    weather_df = df.copy()
    # 소수점 2자리로 포맷팅
    for col in weather_df.select_dtypes(include=['float64', 'float32']):  # float 타입 열에 대해
        weather_df[col] = weather_df[col].map(lambda x: f"{x:.2f}")  # 소수점 2자리로 변환

    return DataTable(
        id='weather-table',
        columns=[
            {"name": "", "id": "row_number"},
            *[{"name": i, "id": i} for i in weather_df.columns]
        ],
        data=[
            {**row, "row_number": index + 1}
            for index, row in weather_df.iterrows()
        ],
        page_size=10,
        style_table={'overflowX': 'auto', 'margin': '20px'},
        style_header={
            'backgroundColor': 'lightgrey',
            'fontWeight': 'bold',
            'textAlign': 'center',
            'padding': '10px'
        },
        style_cell={
            'padding': '10px',
            'textAlign': 'left',
        },
        sort_action='native',
        # filter_action='native',
        # row_selectable='multi',
        # selected_rows=[],
    )


def show_classification_report_table():
    """Classification report 테이블 생성 함수"""
    return html.Div([
        # html.H4("Classification Report"),
        DataTable(
            id='classification-report-table',
            columns=[
                {"name": "Metric", "id": "index"},
                {"name": "Precision", "id": "precision"},
                {"name": "Recall", "id": "recall"},
                {"name": "F1-Score", "id": "f1-score"},
                {"name": "Support", "id": "support"}
            ],
            data=[],  # 초기 데이터는 빈 목록
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'},
            style_header={'backgroundColor': 'lightgrey'},
            style_data={'whiteSpace': 'normal', 'height': 'auto'},
            page_size=10  # 페이지 크기 설정 (선택 사항)
        )
    ], title='classification Report')


def create_layout(df):
    # 레이아웃 정의
    return html.Div([

        # side sticky bar (/assets/icons/... 경로는 404 발생하여, static 경로로 변경)
        html.Div(className='fixed-sidebar', children=[
            html.Ul([
                html.Li(html.A(html.Img(src='/static/icons/home.svg', style={'width': '20px', 'height': '20px'}), href="/")),
                html.Li(html.A(html.Img(src='/static/icons/gauge-high.svg', style={'width': '20px', 'height': '20px'}),
                               href="/dash")),
                html.Li(
                    html.A(html.Img(src='/static/icons/arrow-up.svg', style={'width': '20px', 'height': '20px'}), href="#top",
                           className='dash-scroll-link')),
                html.Li(html.A(html.Img(src='/static/icons/arrow-down.svg', style={'width': '20px', 'height': '20px'}),
                               href="#bottom", className='dash-scroll-link')),
            ]),
        ]),

        html.Div([
            # html.H1("스마트 농업 토양 습도 예측 대시보드"),
            # html.H1("Dashboard"),

            html.Div([
                html.H2("Weather Data"),
                show_weather_data_table(df),
            ], style={'padding': '2% 5% 0 0'}),

            html.Div([
                html.H2("Temperature & Humidity Histogram"),
                create_histogram_graph('histogram-temp', df, 'Temperature', 'Temperature Distribution'),
                create_histogram_graph('histogram-hum', df, 'Humidity', 'Humidity Distribution'),
            ], style={'backgroundColor': '#f8f9fa', 'paddingTop': '5%'}),

            html.Div([
                html.H2("Model Evaluation"),
                html.H4(id='model-evaluation'),
                dcc.Graph(id='confusion-matrix'),
                dcc.Graph(id='scatter-plot'),
                show_classification_report_table(),
            ], style={'paddingTop': '5%', 'marginBottom': '2%'}),
        ], style={'textAlign': 'center'}),

        html.Div([
            html.H2("Predict Soil Moisture", style={'textAlign': 'center'}),
            html.Div([
                html.Div([
                    create_slider('temp-input', "Temperature (°C):", 10, 36, 20, {i: str(i) for i in range(10, 37, 2)}),
                    create_slider('hum-input', "Humidity (%):", 20, 90, 50, {i: str(i) for i in range(20, 91, 10)}),
                    create_slider('rain-input', "Rainfall (mm):", 0, 200, 120, {i: str(i) for i in range(0, 201, 20)}),
                    create_slider('wind-input', "Wind Speed (m/s):", 0, 15, 5, {i: str(i) for i in range(0, 17)}),

                    create_dropdown('soil-type-input', "Soil Type:",
                                    [{'label': soil, 'value': soil} for soil in config.SOIL_TYPES], 'Sandy'),
                    html.Button('예측하기', id='predict-button', className='button', style={
                        'background': 'radial-gradient(circle, rgba(63,94,251,1) 0%, rgba(70,252,216,1) 100%)'}),

                    html.Div(html.H3(id='prediction-output', className='output', style={'padding-bottom': '5%'}))
                ], style={'paddingRight': '5%'}),

            ], style={'width': '80%', 'margin': '0 auto'}),

        ], style={'backgroundColor': '#f8f9fa', 'paddingTop': '5%'}),

    ])
    # ], style={'marginTop': '50px'})

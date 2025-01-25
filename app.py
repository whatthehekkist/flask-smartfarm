import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
# from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import config

import dash
from layout import create_layout
from model import train_model


df = train_model()

# 대시보드 앱 생성
app = dash.Dash(__name__)
app.title = 'Smart Farm'
app.layout = create_layout(df)


# 서버 실행
if __name__ == '__main__':
    app.run_server(debug=True)

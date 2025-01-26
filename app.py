from flask import Flask, render_template, redirect, url_for, request
import dash
from layout import create_layout
from model import train_model

# server
server = Flask(__name__)

# 모델 생성 및 학습
df = train_model()

# Dash 앱 생성
app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')
app.title = 'Smart Farm'
app.layout = create_layout(df)


# route /
@server.route('/')
def index():
    return render_template('index.html')


# route /dash
@server.route('/dash')
def dash():
    return app.index()


# 서버 실행
if __name__ == '__main__':
    server.run(debug=True)

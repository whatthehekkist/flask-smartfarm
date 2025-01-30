# pagination test

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
    # pagination
    page = request.args.get('page', 1, type=int)
    per_page = 10
    total = len(df)
    start = 1 if page < 1 else (page - 1) * per_page
    end = start + per_page
    data = df.iloc[start:end]

    return render_template(
        'index.html',
        data=data.to_html(classes='data', header='true', index=False),
        page=page,
        total=total,
        per_page=per_page
    )


# route /dash
@server.route('/dash')
def dash():
    return app.index()


# 서버 실행
if __name__ == '__main__':
    server.run(debug=True)

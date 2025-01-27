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

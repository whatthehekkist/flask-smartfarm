import dash
import markdown
from markdown_text import markdown_text
from flask import Flask, render_template
from layout import create_layout
from model import train_model

# server
server = Flask(__name__)

# create and train model
df = train_model()

# Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')
app.title = 'Smart Farm'
app.layout = create_layout(df)


# route /
@server.route('/')
def index():
    content = markdown.markdown(markdown_text)
    return render_template('index.html', content=content)


# route /dash
@server.route('/dash')
def dash():
    return app.index()


# run server
if __name__ == '__main__':
    server.run(debug=True)

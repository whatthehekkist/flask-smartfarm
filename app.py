import dash
import markdown
from markdown_text import markdown_text
from flask import Flask, render_template
from layout import create_layout
from model import train_model

# server
app = Flask(__name__)

# create and train model
df = train_model()

# Dash app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')
dash_app.title = 'Smart Farm'
dash_app.layout = create_layout(df)


# route /
@app.route('/')
def index():
    content = markdown.markdown(markdown_text)
    return render_template('index.html', content=content)


# route /dash
@app.route('/dash')
def dash():
    return dash_app.index()


# run server
if __name__ == '__main__':
    app.run(debug=True)  # dev


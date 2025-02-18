# https://flask-smartfarm.onrender.com
ML tutorial that monitors and predicts soil moisture using some random data.
Implemented a logistic regression model to predict whether soil moisture levels are high or low based on data you will create,
Visualize it as a dashboard.

![img](https://github.com/user-attachments/assets/31bc84c4-d84a-4f23-95d1-e73e4e86c68e)


## tech stack and tools
- framework: Flask
- editor: Pycharm community
- git: <a href="https://github.com/whatthehekkist/flask-smartfarm" target="_blank">github.com/whatthehekkist/flask-smartfarm</a>
- deploy: Render
  - Build Command: `pip install -r requirements.txt`
  - Start Command: `gunicorn app:app --bind 0.0.0.0:10000`
  - <a href="https://flask-smartfarm.onrender.com/" target="_blank">flask-smartfarm.onrender.com/</a>: home
  - <a href="https://flask-smartfarm.onrender.com/dash" target="_blank">flask-smartfarm.onrender.com/dash</a>: dashboard
  - note: the deployed app does a cold start so it can take more than 30 sec to get server response
    - <a href="https://render.com/docs/free?_gl=1*s3hlkt*_gcl_au*MjAyMDI2NzYxNy4xNzM5ODM4Mjk3*_ga*MTA2NzE0MDM0OC4xNzM5ODM4Mjk4*_ga_QK9L9QJC5N*MTczOTgzODI5Ny4xLjEuMTczOTgzODM0My4xNC4wLjA.#spinning-down-on-idle" target="_blank">Spinning down on idle</a>
- libraries
  ```python
  pandas~=2.2.3
  numpy~=2.2.2
  dash~=2.18.2
  plotly~=5.24.1
  scikit-learn~=1.6.1
  Flask~=3.0.3
  matplotlib~=3.10.0
  seaborn~=0.13.2
  Markdown~=3.7
  Gunicorn==20.1.0 # for Render deploy
  ```
- blog post: <a href="https://dev-whatthehekkist.netlify.app/project/python/smartfarm/" target="_blank">Flask &#124; Smart Farm Soil Moisture Predictor</a>
 

[//]: # (# resources)

[//]: # (- [home]&#40;home.pdf&#41;)

[//]: # (- [dash]&#40;dash.pdf&#41;)
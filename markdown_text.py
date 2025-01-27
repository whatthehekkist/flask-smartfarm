markdown_text = """
# Smart Farm Soil Moisture Prediction Tutorial
## Objectives
Smart farm can utilize environmental data to monitor and predict soil moisture.
Implement a logistic regression model to predict whether soil moisture levels are high or low based on data you will create,
Visualize it as a dashboard.

## 1: Creating and Exploring Datasets
Create a data frame using pandas and numpy using the data given below.<br>
(The number of data is 1000 each.)

- Temperature (range 10~35°C) => Temperature
- Humidity (20~90% range) => Humidity
- Soil type (Sandy, Clay, Loamy) => Soil variable SoilType, type ['Sandy', 'Clay', 'Loamy']
- Precipitation (range 0~200 mm) => Rainfall
- Wind speed (range 0~15 m/s) => WindSpeed
- Soil humidity (range 10~50%) => SoilMoisture

**requirements**

- Print the first five rows of the generated dataset.
- Check the basic statistics of the data (using '.describe()') and check for missing values.
- Visualize the distribution of 'Temperature', 'Humidity', and 'Rainfall' as a histogram.

<pre>
# collab
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1. 데이터 생성
MAX_SIZE = 1000
soil_types = ['Sandy', 'Clay', 'Loamy']

weather_dict = {
    'Temperature': np.random.uniform(10, 35, MAX_SIZE).tolist(),
    'Humidity': np.random.uniform(20, 90, MAX_SIZE).tolist(),
    'SoilType': np.random.choice(soil_types, size=MAX_SIZE).tolist(),
    'Rainfall': np.random.uniform(0, 200, MAX_SIZE).tolist(),
    'WindSpeed': np.random.uniform(0, 15, MAX_SIZE).tolist(),
    'SoilMoisture': np.random.uniform(10, 50, MAX_SIZE).tolist()
}

df = pd.DataFrame(weather_dict)

# 2. 데이터셋의 처음 5개 행 출력
print("처음 5개 행:\\n", df.head())

# 3. 데이터 기본 통계 확인 및 결측값 점검
print("\\n기본 통계:\\n", df.describe())
print("\\n결측값 여부:\\n", df.isnull().sum())

# 4. 'Temperature', 'Humidity', 'Rainfall'의 분포를 히스토그램으로 시각화
plt.figure(figsize=(13, 6))
for i, col in enumerate(['Temperature', 'Humidity', 'Rainfall']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30, alpha=0.5)
    plt.title(f'{col}')
plt.tight_layout()
plt.show()
</pre>
<pre>
처음 5개 행:
    Temperature   Humidity SoilType    Rainfall  WindSpeed  SoilMoisture
0    26.896728  75.819363    Sandy   82.055911  11.993707     15.994439
1    25.636575  87.113933     Clay   82.771837   0.933470     16.175567
2    23.466401  26.530198     Clay  108.657674  11.153672     25.178655
3    33.327757  80.533965    Loamy   10.011543   0.163955     34.152241
4    19.684246  39.535942    Loamy  125.081637   0.876312     37.257171

기본 통계:
        Temperature     Humidity     Rainfall    WindSpeed  SoilMoisture
count  1000.000000  1000.000000  1000.000000  1000.000000   1000.000000
mean     22.749019    54.811090   100.545332     7.460358     30.530895
std       7.119887    20.601651    57.045995     4.278176     11.781820
min      10.007779    20.042533     0.045427     0.002412     10.056851
25%      16.638168    36.428490    52.048915     3.752343     20.114446
50%      22.760063    55.264473   101.474081     7.467382     30.617053
75%      28.834184    72.671865   152.393464    11.183821     40.865565
max      34.987810    89.684556   199.989511    14.979476     49.982608

결측값 여부:
 Temperature     0
Humidity        0
SoilType        0
Rainfall        0
WindSpeed       0
SoilMoisture    0
dtype: int64
</pre>

![Image](https://github.com/user-attachments/assets/38c158a6-953f-4b9d-a8c2-c062fb823ae2)

## 2: Preprocessing Datasets
dev env

- Flask
- Pycharm community

app.py
<pre>
import dash
from model import train_model

# server
server = Flask(__name__)

# create and train model
df = train_model()

# run server
if __name__ == '__main__':
    server.run(debug=True)
</pre>

config.py
<pre>
import numpy as np

MAX_SIZE = 1000
SOIL_TYPES = ['Sandy', 'Clay', 'Loamy']
TARGET_THRESHOLD = 30  # SoilMoisture 변수를 30% 기준
weather_dict = {
    'Temperature': np.random.uniform(10, 35, MAX_SIZE).tolist(),
    'Humidity': np.random.uniform(20, 90, MAX_SIZE).tolist(),
    'SoilType': np.random.choice(SOIL_TYPES, size=MAX_SIZE).tolist(),
    'Rainfall': np.random.uniform(0, 200, MAX_SIZE).tolist(),
    'WindSpeed': np.random.uniform(0, 15, MAX_SIZE).tolist(),
    'SoilMoisture': np.random.uniform(10, 50, MAX_SIZE).tolist()
}
</pre>

model.py
<pre>
# libraries
import config
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import NearestNeighbors

import dash
from dash import callback
from dash.dependencies import Input, Output

import plotly.express as px

# global variables
df = None
model = None
X_train = None
X_test = None
y_train = None
y_test = None
</pre>

**requirements**

- Convert a categorical variable, such as SoilType, to a number by one-hot encoding it.
- Binary classify the SoilMoisture variable based on 30% (0: low, 1: high).
- Split the data into a training set (80%) and a test set (20%).
- Perform data normalization and print the mean and standard deviation.

train_model() in model.py
<pre>
def train_model():
    global df, model, X_train, X_test, y_train, y_test
    df = pd.DataFrame(config.weather_dict)
    # print(df.head())
    # print(df.describe())

    # SoilMoisture 이진 분류 생성
    df['SoilMoistureDegree'] = np.where(df['SoilMoisture'] > config.TARGET_THRESHOLD, 1, 0)

    # 특성과 타겟 분리
    X = df.drop(columns=['SoilMoisture', 'SoilMoistureDegree'])
    y = df['SoilMoistureDegree']

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 토양 유형(SoilType) 원-핫 인코딩 및 표준화 스케일링을 포함하는 파이프라인 정의
    # OneHotEncoder를 사용하여 훈련 세트와 테스트 세트에 대해 동일한 인코딩을 적용
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Temperature', 'Humidity', 'Rainfall', 'WindSpeed']),
            ('cat', OneHotEncoder(drop='first'), ['SoilType'])
        ]
    )

    # 모델 파이프라인 정의 (로지스틱회귀)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 모델 학습
    model.fit(X_train, y_train)

    # 학습된 모델 출력
    # 1. 전처리기(Preprocessor) 가져오기
    preprocessor = model.named_steps['preprocessor']

    # 2. 수치형 변수의 평균과 표준편차 출력
    if hasattr(preprocessor.named_transformers_['num'], 'mean_'):
        print("수치형 변수 평균:", preprocessor.named_transformers_['num'].mean_)
        print("수치형 변수 표준편차:", preprocessor.named_transformers_['num'].scale_)

    # 3. 원-핫 인코더의 변환된 클래스 출력
    if hasattr(preprocessor.named_transformers_['cat'], 'categories_'):
        print("원-핫 인코딩된 클래스:", preprocessor.named_transformers_['cat'].categories_)

    .
    .
    .
    return df
</pre>
<pre>
수치형 변수 평균: [ 22.64529429  55.32044726 101.57732775   7.47171436]
수치형 변수 표준편차: [ 7.18744874 20.95467849 57.21062832  4.20746486]
원-핫 인코딩된 클래스: [array(['Clay', 'Loamy', 'Sandy'], dtype=object)]
</pre>

## 3. Implement Logistic Regression Model

**requirements**

- Train the model using LogisticRegression. (Random seed: 42, maximum number of iterations: 1000)
- Make predictions using test data.
- Print the model’s accuracy.

train_model() in model.py
<pre>
def train_model():
    .
    .
    .
    # 모델 파이프라인 정의 (로지스틱회귀)
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # 모델 학습
    model.fit(X_train, y_train)

    # 학습된 모델 출력
    .
    .
    .
    # 4. 모델의 계수 및 정확도 출력 (Logistic Regression)
    model_coefficients = model.named_steps['classifier'].coef_
    model_intercept = model.named_steps['classifier'].intercept_

    print("모델 계수:", model_coefficients)
    print("모델 절편:", model_intercept)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print("모델 정확도:", accuracy)

    return df
</pre>
<pre>
모델 계수: [[-0.01166415 -0.02955629 -0.01722798 -0.07567536 -0.2606666  -0.03820069]]
모델 절편: [0.02364861]
모델 정확도: 0.485  # avg 0.45 to 0.55
</pre>

## [DO FROM THIS] 4.Model evaluation and performance analysis
**requirements**

- Create and visualize a confusion matrix.
- Calculate precision, recall, and F1-score.
- Create a scatterplot comparing actual and predicted values.

"""
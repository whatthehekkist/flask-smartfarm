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
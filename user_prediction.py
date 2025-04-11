import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import radians, sin, cos, sqrt, atan2
import time

def train_user_prediction_model(data_path):
    print('👉 Training XGBoost model...')
    ################ feature extract ######################
    data = pd.read_csv(data_path)
    data['start_x'] = data['start_point'].str.extract(r'POINT\(([-.\d]+) ([-.\d]+)\)')[0].astype(float)
    data['start_y'] = data['start_point'].str.extract(r'POINT\(([-.\d]+) ([-.\d]+)\)')[1].astype(float)
    data['end_x'] = data['end_point'].str.extract(r'POINT\(([-.\d]+) ([-.\d]+)\)')[0].astype(float)
    data['end_y'] = data['end_point'].str.extract(r'POINT\(([-.\d]+) ([-.\d]+)\)')[1].astype(float)

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek

    X = data[['start_x', 'start_y', 'hour', 'day_of_week']]  # Fixed 'Dallas' typo
    y = data[['end_x', 'end_y']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    ############################ Train ############################
    xgb = XGBRegressor(n_estimators=100, max_depth=20, learning_rate=0.1, random_state=42, n_jobs=-1)
    print('👉 start training...') 
    start_time = time.time()
    xgb.fit(X_train, y_train)
    end_time = time.time()
    print(f'👉 Model training completed in {end_time - start_time:.2f} seconds')
    print('👉 Generating predictions...')
    # make predictions
    y_pred = xgb.predict(X_test)
    print('👉 Prediction done!')
    ############################ Evaluation ############################
    mse = mean_squared_error(y_test, y_pred)
    print(f'👉 Model MSE: {mse}')

    distances = [haversine(y_test.iloc[i, 0], y_test.iloc[i, 1], y_pred[i, 0], y_pred[i, 1])
                 for i in range(len(y_test))]
    mean_distance = sum(distances) / len(distances)
    print(f'👉 Model Mean Distance Error: {mean_distance:.2f} km')
    print('👉 Model training and evaluation completed!')
    return xgb


# a more precise way to calculate distance error
def haversine(lon1, lat1, lon2, lat2):
    R = 6371
    dlon = radians(lon2 - lon1)
    dlat = radians(lat2 - lat1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

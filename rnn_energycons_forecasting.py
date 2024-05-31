import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Загрузка и объединение данных
energy_data = pd.read_csv('energy_data.csv', parse_dates=['timestamp'])
weather_data = pd.read_csv('weather_data.csv', parse_dates=['timestamp'])

# Приведение данных к часовому шагу
energy_data = energy_data.set_index('timestamp').resample('H').mean().reset_index()
weather_data = weather_data.set_index('timestamp').resample('H').interpolate().reset_index()

# Объединение данных
data = pd.merge(energy_data, weather_data, on=['timestamp', 'region'])

# Создание признаков
data['hour'] = data['timestamp'].dt.hour
data['day_of_week'] = data['timestamp'].dt.dayofweek
data['month'] = data['timestamp'].dt.month

data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# Лаговые признаки
for lag in [1, 24, 168]:  # за прошлый час, день, неделю
    data[f'lag_{lag}'] = data['energy_consumption'].shift(lag)

data = data.dropna()

# Нормализация данных
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop(['timestamp', 'region'], axis=1))

# Формирование обучающей и тестовой выборок
X = scaled_data[:, 1:]  # все, кроме потребления
y = scaled_data[:, 0]   # потребление

# Разделение на обучающую и тестовую выборки
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Формирование данных для RNN
def create_rnn_data(X, y, time_steps=24):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 24
X_train_rnn, y_train_rnn = create_rnn_data(X_train, y_train, time_steps)
X_test_rnn, y_test_rnn = create_rnn_data(X_test, y_test, time_steps)

# Построение модели RNN
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_steps, X_train_rnn.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Обучение модели
early_stop = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(X_train_rnn, y_train_rnn, epochs=100, validation_split=0.2, callbacks=[early_stop])

# Оценка модели
loss = model.evaluate(X_test_rnn, y_test_rnn)
print(f'Loss: {loss}')

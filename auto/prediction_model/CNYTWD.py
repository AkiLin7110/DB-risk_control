# Import necessary libraries for data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tabulate import tabulate
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("fivethirtyeight")

# For reading stock data from yahoo
from pandas_datareader.data import DataReader
import yfinance as yf
from pandas_datareader import data as pdr
# For time stamps
from datetime import datetime

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam, SGD

import random
import numpy as np
import tensorflow as tf

# 固定隨機種子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# 讀檔案
df = pd.read_excel('./FX_CNY_FR.xlsx')
data = df.filter(['即期買入'])
dataset = data.values

BATCH_SIZE = 90
PREVIOUS = 60
PREDICT = 15

# Split the dataset into training, validation, and test sets
training_data_len = int(np.ceil(len(dataset) * 0.70))
validation_data_len = int(np.ceil(len(dataset) * 0.85))

train_data = dataset[:training_data_len, :]
val_data = dataset[training_data_len:validation_data_len, :]
test_data = dataset[validation_data_len:, :]

# Scale the data using MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Apply scaling to the training, validation, and test data
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data)
scaled_test_data = scaler.transform(test_data)

# Verify scaling
print("Data scaling completed.")

# Create the training data set
x_train, y_train = [], []


for i in range(PREVIOUS, len(scaled_train_data)-PREDICT):
    x_train.append(scaled_train_data[i-PREVIOUS:i, 0])
    y_train.append(scaled_train_data[i+PREDICT, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Preparing validation and test data
x_val, y_val = [], []
x_test, y_test = [], []

for i in range(PREVIOUS, len(scaled_val_data)-PREDICT):
    x_val.append(scaled_val_data[i-PREVIOUS:i, 0])
    y_val.append(scaled_val_data[i+PREDICT, 0])

for i in range(PREVIOUS, len(scaled_test_data)-PREDICT):
    x_test.append(scaled_test_data[i-PREVIOUS:i, 0])
    y_test.append(scaled_test_data[i+PREDICT, 0])

x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

# Reshape the validation and test sets
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print("Data preparation completed.")



'''Model1: LSTM-NAG'''
# Function to build the model
def build_model(model_type, optimizer):
    model = Sequential()
    if model_type == 'LSTM':
        model.add(LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(64, return_sequences=False))
    elif model_type == 'GRU':
        model.add(GRU(128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(GRU(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    print("Model building function defined.")
    return model

# Makes predictions with each model
def make_predictions(model, x_data):
    predictions = model.predict(x_data)
    return scaler.inverse_transform(predictions)

def plot_predictions(model, x_train, x_test, scaler, data, training_data_len, validation_data_len ,title=None):
    # Get model predictions
    train_predictions = model.predict(x_train)
    test_predictions = model.predict(x_test)
    
    # Inverse transform predictions
    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    
    # Create training plot
    plt.figure(figsize=(15,6))
    plt.subplot(1, 2, 1)
    plt.title(title+' training dataset')
    plt.plot(data[PREVIOUS+PREDICT:training_data_len], label='Actual Price', color='black',linewidth=1)
    plt.plot(data.index[PREVIOUS+PREDICT:training_data_len], train_predictions, label='Predict Price', color='blue',linewidth=1)
    plt.xlabel('Days')
    plt.ylabel('TWD')
    plt.legend()
    
    # Create testing plot
    plt.subplot(1, 2, 2)
    plt.title(title+' testing dataset')
    test_data = data[validation_data_len:]
    plt.plot(test_data, label='Actual Price', color='black',linewidth=1)
    plt.plot(test_data.index[PREVIOUS+PREDICT:], test_predictions, label='Predict Price', color='blue',linewidth=1)
    plt.xlabel('Days')
    plt.ylabel('TWD')
    plt.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

def calculate_direction_accuracy(y_true, y_pred):
    """
    Calculate the accuracy of predicted price movement directions.
    
    Parameters:
    y_true: numpy array of actual prices
    y_pred: numpy array of predicted prices
    
    Returns:
    float: accuracy percentage
    dict: detailed metrics including total predictions and correct ones
    """
    # Calculate actual and predicted directions
    actual_direction = np.sign(np.diff(y_true.flatten(), n = PREDICT))
    pred_direction = np.sign(np.diff(y_pred.flatten(), n = PREDICT))
    
    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == pred_direction)
    total_predictions = len(actual_direction)
    accuracy = (correct_predictions / total_predictions) * 100
    
    # Calculate separate accuracies for up and down movements
    actual_ups = actual_direction == 1
    actual_downs = actual_direction == -1
    
    up_accuracy = np.sum((actual_direction == pred_direction) & actual_ups) / np.sum(actual_ups) * 100
    down_accuracy = np.sum((actual_direction == pred_direction) & actual_downs) / np.sum(actual_downs) * 100
    
    metrics = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': accuracy,
        'up_movement_accuracy': up_accuracy,
        'down_movement_accuracy': down_accuracy
    }
    
    return accuracy, metrics


'''Model2: LSTM-GAN'''
class LSTMMonteCarloPredictor:
    def __init__(self, n_simulations=1000):
        self.n_simulations = n_simulations
        self.model = None
        self.scaler = MinMaxScaler()
        self.price_scaler = MinMaxScaler()
        
    def build_model(self, n_steps, n_features, params):
        """建立雙向LSTM模型"""
        model = Sequential()
        model.add(LSTM(params['layer1'], activation='relu', 
                                   return_sequences=True), 
                                   )
        model.add(LSTM(params['layer2'], activation='relu', 
                                   return_sequences=True))
        model.add(LSTM(params['layer3'], activation='relu',
                                   return_sequences=False))
        # model.add(Bidirectional(LSTM(params['layer1'], activation='relu', 
        #                            return_sequences=True), 
        #                            input_shape=(n_steps, n_features)))
        # model.add(Bidirectional(LSTM(params['layer2'], activation='relu', 
        #                            return_sequences=True)))
        # model.add(Bidirectional(LSTM(params['layer3'], activation='relu',
        #                            return_sequences=False)))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        self.model = model

    def monte_carlo_simulation(self, last_price, n_days, volatility, n_sims=None):
        """執行蒙特卡羅模擬"""
        if n_sims is None:
            n_sims = self.n_simulations
        
        dt = 1/252  # 交易日
        mu = 0  # 假設漂移為0
        
        # 生成價格路徑
        simulations = np.zeros((n_sims, n_days))
        for i in range(n_sims):
            # 生成隨機游走
            random_walk = np.random.normal(
                (mu - 0.5 * volatility**2) * dt,
                volatility * np.sqrt(dt),
                n_days
            )
            
            # 計算累積收益
            cumulative_returns = np.exp(np.cumsum(random_walk))
            
            # 生成價格路徑
            simulations[i] = last_price * cumulative_returns
        
        return simulations

    def predict_with_monte_carlo(self, X_test_reshaped, last_price, n_days=5):
        """結合LSTM預測和蒙特卡羅模擬"""
        # 獲取LSTM預測
        lstm_pred_scaled = self.model.predict(X_test_reshaped)
        lstm_pred = self.price_scaler.inverse_transform(lstm_pred_scaled)
        
        # 使用預測值計算波動率
        predicted_prices = lstm_pred.flatten()
        returns = np.diff(predicted_prices) / predicted_prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)
        
        # 執行蒙特卡羅模擬
        mc_sims = self.monte_carlo_simulation(last_price, n_days, volatility)
        
        # 計算預測區間
        confidence_intervals = np.percentile(mc_sims, [5, 95], axis=0)
        mean_prediction = np.mean(mc_sims, axis=0)
        
        return {
            'lstm_prediction': lstm_pred,
            'monte_carlo_mean': mean_prediction,
            'lower_bound': confidence_intervals[0],
            'upper_bound': confidence_intervals[1]
        }

# LSTM with Nesterov
optimizer_nag = SGD(momentum=0.9, nesterov=True)
model_lstm_nag = build_model('LSTM', optimizer_nag)
history_lstm_nag = model_lstm_nag.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=BATCH_SIZE, epochs=100, verbose=1)
print("LSTM with Nesterov training completed.")

# Generate predictions and plot
plot_predictions(model_lstm_nag, x_train, x_test, scaler, data, training_data_len, validation_data_len , title='LSTM with NAG')


predictor = LSTMMonteCarloPredictor()

# 準備LSTM數據
n_steps = 1
n_features = x_train.shape[1]
X_train_reshaped = x_train.reshape((x_train.shape[0], n_steps, n_features))
X_test_reshaped = x_test.reshape((x_test.shape[0], n_steps, n_features))

# 模型參數
params = {
    'layer1': 259,
    'layer2': 410,
    'layer3': 473,
    'epochs': 100
}

# 建立並訓練模型
predictor.build_model(n_steps, n_features, params)
history = predictor.model.fit(
    X_train_reshaped, 
    y_train,
    epochs=params['epochs'],
    batch_size = BATCH_SIZE,
    # validation_split = 0.2,
    verbose = 1
)

# 獲取LSTM預測結果
train_pred_scaled = predictor.model.predict(X_train_reshaped)
test_pred_scaled = predictor.model.predict(X_test_reshaped)

# 轉換回原始比例
# Fit the scaler on y_train data
predictor.price_scaler.fit(y_train.reshape(-1, 1))
train_pred = predictor.price_scaler.inverse_transform(train_pred_scaled)
test_pred = predictor.price_scaler.inverse_transform(test_pred_scaled)
# y_train_orig = predictor.price_scaler.inverse_transform(y_train)
# y_test_orig = predictor.price_scaler.inverse_transform(y_test)

# Example usage for each model:
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

# # Calculate accuracy for each model
pred_lstm_nag = make_predictions(model_lstm_nag, x_test)
lstm_nag_acc, lstm_nag_metrics = calculate_direction_accuracy(y_true, pred_lstm_nag)
print(f"LSTM-NAG Direction Accuracy: {lstm_nag_acc:.2f}%")

lstm_nag_acc, lstm_nag_metrics = calculate_direction_accuracy(y_true, test_pred)
print(f"LSTM-GAN Direction Accuracy: {lstm_nag_acc:.2f}%")

# 儲存 LSTM-NAG 模型
model_lstm_nag.save("lstm_nag_model.h5")
print("LSTM-NAG 模型已儲存為 lstm_nag_model.h5")

# 儲存 LSTM-GAN 模型
predictor.model.save("lstm_gan_model.h5")
print("LSTM-GAN 模型已儲存為 lstm_gan_model.h5")

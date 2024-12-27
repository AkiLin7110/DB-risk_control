from comet_ml import Experiment
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam, SGD


from keras.models import load_model


# 讀檔案
df = pd.read_excel('./FX_CNY_FR.xlsx')
data = df.filter(['即期買入'])
dataset = data.values

BATCH_SIZE = 90
PREVIOUS = 60
PREDICT = 12

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
    
    up_up = np.sum((actual_direction == pred_direction) & actual_ups)
    up_down =  np.sum((actual_direction != pred_direction) & actual_ups)
    down_down = np.sum((actual_direction == pred_direction) & actual_downs)
    down_up = np.sum((actual_direction != pred_direction) & actual_downs)

    metrics = {
        'total_predictions': total_predictions,
        'correct_predictions': correct_predictions,
        'overall_accuracy': accuracy,
        'up_movement_accuracy': up_accuracy,
        'down_movement_accuracy': down_accuracy,
        'up_up':up_up,
        'up_down':up_down,
        'down_down':down_down,
        'down_up':down_up
    }
    
    return accuracy, metrics

# Makes predictions with each model
def make_predictions(model, x_data):
    predictions = model.predict(x_data)
    return scaler.inverse_transform(predictions)

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
print(f'x_val.shape:{x_val.shape}')
print("Data preparation completed.")

# 載入 LSTM-NAG 模型
model_lstm_nag = load_model("lstm_nag_model.h5")
print("LSTM-NAG 模型載入成功")


# Example usage for each model:
y_true = scaler.inverse_transform(y_test.reshape(-1, 1))


# Calculate accuracy for each model
pred_lstm_nag = make_predictions(model_lstm_nag, x_test)
lstm_nag_acc, lstm_nag_metrics = calculate_direction_accuracy(y_true, pred_lstm_nag)
print(f"LSTM-NAG Direction Accuracy: {lstm_nag_acc:.2f}%")

# Last Prediction
x_wanted_prev = scaled_test_data[-PREVIOUS-PREDICT:-PREDICT, 0]
x_wanted_prev = np.array(x_wanted_prev)
x_wanted_prev = np.reshape(x_wanted_prev, (1, x_wanted_prev.shape[0], 1))
x_wanted_prev = make_predictions(model_lstm_nag, x_wanted_prev)

x_wanted = scaled_test_data[-PREVIOUS:, 0]
x_wanted = np.array(x_wanted)
x_wanted = np.reshape(x_wanted, (1, x_wanted.shape[0], 1))
x_wanted = make_predictions(model_lstm_nag, x_wanted)

predict_direction = (x_wanted - x_wanted_prev)
if predict_direction >= 0:
    result = f'未來{PREDICT}日, 基準價位:{x_wanted_prev[0][0]}, 可能上漲{predict_direction[0][0]}!'
else:
    predict_direction = -predict_direction
    result = f'未來{PREDICT}日, 基準價位:{x_wanted_prev[0][0]}, 可能下跌{predict_direction[0][0]}!'

print(result, f';上升/下跌準確率為:{lstm_nag_acc:.2f}%; 上升準確率為:{lstm_nag_metrics}')
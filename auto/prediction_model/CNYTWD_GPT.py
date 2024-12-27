from comet_ml import Experiment
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from keras.optimizers import Adam, SGD

# Import and setup
BATCH_SIZE = 2500
PREVIOUS = 60
PREDICT = 20

# Load data
df = pd.read_excel('./FX_CNY_FR.xlsx')
data = df.filter(['即期買入'])  # Replace with appropriate column name
dataset = data.values

# Split the dataset
training_data_len = int(np.ceil(len(dataset) * 0.70))
validation_data_len = int(np.ceil(len(dataset) * 0.85))

train_data = dataset[:training_data_len, :]
val_data = dataset[training_data_len:validation_data_len, :]
test_data = dataset[validation_data_len:, :]

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data)
scaled_val_data = scaler.transform(val_data)
scaled_test_data = scaler.transform(test_data)

# Prepare training data
x_train, y_train = [], []
for i in range(PREVIOUS, len(scaled_train_data) - PREDICT):
    x_train.append(scaled_train_data[i-PREVIOUS:i, 0])
    y_train.append(scaled_train_data[i+PREDICT, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Prepare validation and test data
x_val, y_val, x_test, y_test = [], [], [], []
for i in range(PREVIOUS, len(scaled_val_data) - PREDICT):
    x_val.append(scaled_val_data[i-PREVIOUS:i, 0])
    y_val.append(scaled_val_data[i+PREDICT, 0])

for i in range(PREVIOUS, len(scaled_test_data) - PREDICT):
    x_test.append(scaled_test_data[i-PREVIOUS:i, 0])
    y_test.append(scaled_test_data[i+PREDICT, 0])

x_val, y_val = np.array(x_val), np.array(y_val)
x_test, y_test = np.array(x_test), np.array(y_test)

x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Function to build models
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
    return model

# Function to calculate accuracy
def calculate_direction_accuracy(y_true, y_pred):
    true_directions = np.sign(np.diff(y_true.flatten()))
    pred_directions = np.sign(np.diff(y_pred.flatten()))
    correct = np.sum(true_directions == pred_directions)
    return correct / len(true_directions) * 100

# Initialize Comet experiment
experiment = Experiment(
    api_key="0UMQ6h20m4Jv5QIHCPPuHkvp7",  # Replace with your Comet API key
    project_name="syntec_v2",
    workspace="akilin7110"  # Replace with your Comet workspace
)

# Log shared parameters
experiment.log_parameters({
    "batch_size": BATCH_SIZE,
    "previous_steps": PREVIOUS,
    "predict_steps": PREDICT
})

# Train LSTM-NAG model
optimizer_nag = SGD(momentum=0.9, nesterov=True)
model_lstm_nag = build_model('LSTM', optimizer_nag)
history_lstm_nag = model_lstm_nag.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    batch_size=BATCH_SIZE,
    epochs=1000,
    verbose=1
)

# Log LSTM-NAG metrics
experiment.set_name("LSTM-NAG")
experiment.log_metric("final_train_loss", history_lstm_nag.history['loss'][-1])
experiment.log_metric("final_val_loss", history_lstm_nag.history['val_loss'][-1])

# Calculate and log direction accuracy
pred_lstm_nag = model_lstm_nag.predict(x_test)
accuracy_lstm_nag = calculate_direction_accuracy(scaler.inverse_transform(y_test.reshape(-1, 1)), scaler.inverse_transform(pred_lstm_nag))
experiment.log_metric("direction_accuracy", accuracy_lstm_nag)

# Plot and log LSTM-NAG loss curve
plt.figure()
plt.plot(history_lstm_nag.history['loss'], label="Train Loss")
plt.plot(history_lstm_nag.history['val_loss'], label="Validation Loss")
plt.legend()
plt.title("LSTM-NAG Loss")
plt.savefig("lstm_nag_loss.png")
experiment.log_image("lstm_nag_loss.png")

# # Train LSTM-GAN model
# params = {"layer1": 259, "layer2": 410, "layer3": 473, "epochs": 7}
# model_lstm_gan = Sequential()
# model_lstm_gan.add(LSTM(params['layer1'], activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
# model_lstm_gan.add(LSTM(params['layer2'], activation='relu', return_sequences=True))
# model_lstm_gan.add(LSTM(params['layer3'], activation='relu', return_sequences=False))
# model_lstm_gan.add(Dense(1))
# model_lstm_gan.compile(optimizer='adam', loss='mse')

# history_lstm_gan = model_lstm_gan.fit(
#     x_train, y_train,
#     validation_data=(x_val, y_val),
#     batch_size=128,
#     epochs=params['epochs'],
#     verbose=1
# )

# # Log LSTM-GAN metrics
# experiment.set_name("LSTM-GAN")
# experiment.log_metric("final_train_loss", history_lstm_gan.history['loss'][-1])
# experiment.log_metric("final_val_loss", history_lstm_gan.history['val_loss'][-1])

# # Calculate and log direction accuracy
# pred_lstm_gan = model_lstm_gan.predict(x_test)
# accuracy_lstm_gan = calculate_direction_accuracy(scaler.inverse_transform(y_test.reshape(-1, 1)), scaler.inverse_transform(pred_lstm_gan))
# experiment.log_metric("direction_accuracy", accuracy_lstm_gan)

# # Plot and log LSTM-GAN loss curve
# plt.figure()
# plt.plot(history_lstm_gan.history['loss'], label="Train Loss")
# plt.plot(history_lstm_gan.history['val_loss'], label="Validation Loss")
# plt.legend()
# plt.title("LSTM-GAN Loss")
# plt.savefig("lstm_gan_loss.png")
# experiment.log_image("lstm_gan_loss.png")

# End Comet experiment
experiment.end()

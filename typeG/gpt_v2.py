import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Constants
MOVING_WIN_SIZE = 35
DS_SPLIT = 0.8

# Load data
typeg_data = pd.read_csv("./typeG/data/2330_TW.csv")

# Preprocess data
typeg_data = typeg_data.rename(columns={'Close': 'clo'})
test_day = typeg_data[['clo']].reset_index(drop=True)

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_prices = scaler.fit_transform(test_day.values)

# Prepare dataset for training
all_x, all_y = [], []
for i in range(len(scaler_prices) - MOVING_WIN_SIZE):
    x = scaler_prices[i:i + MOVING_WIN_SIZE]
    y = scaler_prices[i + MOVING_WIN_SIZE]
    all_x.append(x)
    all_y.append(y)
all_x, all_y = np.array(all_x), np.array(all_y)

# Split dataset into training and testing sets
train_ds_size = round(all_x.shape[0] * DS_SPLIT)
train_x, train_y = all_x[:train_ds_size], all_y[:train_ds_size]
test_x, test_y = all_x[train_ds_size:], all_y[train_ds_size:]

# Load and compile the model
reload_model = tf.keras.models.load_model('./typeG/tsmc_v1.h5')
reload_model.compile(optimizer="adam", loss="mean_squared_error")

# Set up callbacks: early stopping and learning rate scheduler
early_stopping = EarlyStopping(monitor="val_loss", patience=150, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=30, min_lr=0.000001, verbose=1)

# Train the model with callbacks
reload_model.fit(train_x, train_y, validation_split=0.21, epochs=1000, callbacks=[early_stopping, reduce_lr])

# Make predictions
preds = scaler.inverse_transform(reload_model.predict(test_x))

# Prepare test data for evaluation
train_data = test_day[:train_ds_size + MOVING_WIN_SIZE]
test_data = test_day[train_ds_size + MOVING_WIN_SIZE:]
test_data = test_data.assign(Predict=preds)

# Calculate shifted values for comparison
test_data = test_data.assign(Shifted=test_data['clo'].shift(1))
test_data.iat[0, -1] = train_data.iat[-1, -1]

# Calculate RMSE and CVRMSE
predict_rmse = mean_squared_error(test_data['clo'], test_data['Predict'], squared=False)
predict_cvrmse = (predict_rmse / test_data['clo'].mean()) * 100
shifted_rmse = mean_squared_error(test_data['clo'], test_data['Shifted'], squared=False)
shifted_cvrmse = (shifted_rmse / test_data['clo'].mean()) * 100

# Print results
print(f"Predict RMSE: {predict_rmse}")
print(f"Predict CVRMSE: {predict_cvrmse}%")
print(f"Shifted RMSE: {shifted_rmse}")
print(f"Shifted CVRMSE: {shifted_cvrmse}%")

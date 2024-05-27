from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

import tensorflow as tf
import numpy as  np
import pandas as pd


MOVING_WIN_SIZE=35
DS_SPLIT=0.8

typeg_data=pd.read_csv("./typeG/data/2330_TW.csv")


# print(os.getcwd())

test_10day=typeg_data.rename(columns={'Close':'clo'})
test_day=test_10day.filter(['clo'])
test_day=test_day.reset_index(drop=True)

scaler=MinMaxScaler(feature_range=(0,1))
scaler_prices=scaler.fit_transform(test_day.values)

all_x,all_y=[],[]
for i in range(len(scaler_prices)-MOVING_WIN_SIZE):
    x=scaler_prices[i:i+MOVING_WIN_SIZE]
    y=scaler_prices[i+MOVING_WIN_SIZE]
    all_x.append(x)
    all_y.append(y)
all_x,all_y=np.array(all_x),np.array(all_y)

train_ds_size=round(all_x.shape[0]*DS_SPLIT)
train_x,train_y=all_x[:train_ds_size],all_y[:train_ds_size]
test_x,test_y=all_x[train_ds_size:],all_y[train_ds_size:]

#import & train
reload_model = tf.keras.models.load_model('./typeG/tsmc_v1.h5')
reload_model.compile(optimizer="adam",loss="mean_squared_error")
callback=EarlyStopping( monitor="val_loss",patience=100,restore_best_weights=True)
reload_model.fit(train_x , train_y , validation_split=0.21 , callbacks=[callback] , epochs=1000)


preds=scaler.inverse_transform(reload_model.predict(test_x))
train_data=test_day[:train_ds_size+MOVING_WIN_SIZE]
test_data=test_day[train_ds_size+MOVING_WIN_SIZE:]
test_data=test_data.assign(Predict=preds)


test_data=test_data.assign(Shifted=test_data['clo'].shift(1))
test_data.iat[0,-1]=train_data.iat[-1,-1]
predict_rmse=mean_squared_error(test_data['clo'],test_data['Predict'],squared=False)
predict_cvrmse=predict_rmse / test_data['clo'].mean()*100
shifted_rmse=mean_squared_error(test_data['clo'],test_data['Shifted'],squared=False)
shifted_cvrmse=shifted_rmse / test_data['clo'].mean()*100


print(f"predict\t=  {predict_cvrmse} % \nshift\t=  {shifted_cvrmse} %")
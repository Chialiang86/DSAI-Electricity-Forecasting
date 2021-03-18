
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import time
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
# You can write code above the if-main block.

def readCSV():
    df = pd.read_csv('20190101-20201031.csv', usecols=['備轉容量(MW)','淨尖峰供電能力(MW)', '尖峰負載(MW)', '備轉容量率(%)'])
    df = pd.DataFrame(df.values, columns=['備轉容量(MW)','淨尖峰供電能力(MW)', '尖峰負載(MW)', '備轉容量率(%)'])
    date_format = pd.read_csv('20190101-20201031.csv', usecols=['日期'])
    date_format = pd.to_datetime(date_format['日期'], yearfirst=True, format='%Y%m%d')
    df['month'] = date_format.dt.month
    df['day'] = date_format.dt.day
    df['dayOfWeek'] = date_format.dt.dayofweek
    return df

def normalize(data):
    data_norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data_norm

def setTrainTestData(data, ref_day, predict_day, ratio):
    x, y = [], []
    for i in range(len(data) - ref_day - predict_day):
        x.append(np.array(data.iloc[i:i + ref_day]))
        y.append(np.array(data.iloc[i + ref_day:i + ref_day + predict_day]['備轉容量(MW)']))
    x,y = np.array(x), np.array(y)
    x_train = x[:int(x.shape[0] * ratio)]
    x_test = x[int(x.shape[0] * ratio):]
    y_train = y[:int(y.shape[0] * ratio)]
    y_test = y[int(y.shape[0] * ratio):]
    return x_train, x_test, y_train, y_test

def shuffle(x, y):
    np.random.seed(int(time.time()))
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

def buildManyToManyModel(shape, batch_size):
  model = Sequential()
  model.add(LSTM(50, input_shape=(shape[1], shape[2]), return_sequences = True))
  model.add(Dropout(0.2))
  model.add(LSTM(50, return_sequences=True))
  # Adding a second LSTM layer and some Dropout regularisation
  model.add(Dropout(0.2))
  model.add(LSTM(50, return_sequences=True))
  
  # Adding a third LSTM layer and some Dropout regularisation
  model.add(LSTM(units = 50, return_sequences = True))
  model.add(Dropout(0.2))
  
  # Adding a fourth LSTM layer and some Dropout regularisation
  model.add(TimeDistributed(Dense(units = 1)))
  model.add(Flatten())
  model.add(Dense(5, activation='linear'))
  model.add(Dense(7))
  model.compile(loss="mean_absolute_error", optimizer="adam", metrics=['mean_absolute_error'])
  model.summary()
  return model

batch_size = 128
ref_day = 30
predict_day = 7
train_ratio = 0.9
epoch = 100

train_data = readCSV()
data_norm = normalize(train_data)
x_train, x_test, y_train, y_test = setTrainTestData(data_norm, ref_day, predict_day, train_ratio)
# y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
# y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))
x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)
for line in x_test:
    print(line)
print('---------')
for line in y_test:
    print(line)

model = buildManyToManyModel(x_train.shape, batch_size)
filepath = "weights-{epoch:02d}-{mean_absolute_error:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='mean_absolute_error', verbose=1, save_best_only=True, mode='min')
history = model.fit(x_train, y_train, verbose=1, callbacks=[checkpoint],\
    validation_data=(x_test, y_test), batch_size=batch_size, epochs=epoch)

if __name__ == '__main__':
    # You should not modify this part, but additional arguments are allowed.
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # The following part is an example.
    # You can modify it at will.
    # df_training = pd.read_csv(args.training)
    # model = Model()
    # model.train(df_training)
    # df_result = model.predict(n_step=7)
    # df_result.to_csv(args.output, index=0)

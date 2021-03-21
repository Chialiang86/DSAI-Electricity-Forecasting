
from numpy.lib.npyio import save
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

# raw data loading
def readCSV(fname):
    df = pd.read_csv(fname, usecols=['淨尖峰供電能力(MW)', '尖峰負載(MW)', '備轉容量(MW)', '備轉容量率(%)', 'temp_high', 'temp_low'])
    date_format = pd.read_csv(fname, usecols=['日期'])
    for i in range(date_format.__len__()):
        day = date_format.values[i]
        l = str(day[0]).split('/')
        # formating the "日期" column (ex: 2021/1/1 -> 20210101)
        src = day if len(l) == 1 else '{}{:02d}{:02d}'.format(int(l[0]), int(l[1]), int(l[2]))
        date_format.values[i] = src
    date_format = pd.to_datetime(date_format['日期'], yearfirst=True, format='%Y%m%d')
    df['month'] = date_format.dt.month
    df['day'] = date_format.dt.day
    df['dayOfWeek'] = date_format.dt.dayofweek
    return df

# normalize data to [0,1]
def normalize(data):
    data_norm = data.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
    return data_norm

# use the number of ref_day's data to predict the number of predict_day's data
# split training/testing data by 'ratio' parameter
def setTrainTestData(data, ref_day, predict_day, ratio=0.9):
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

# use the latest ref_day's data as the input data of 7 days(2021/03/23 - 2021/03/29) prediction
def setPredictInput(data, ref_day, min, range):
    x = []
    x.append(np.array(data.iloc[len(data) - ref_day:]))
    data = data[len(data) - ref_day:]['備轉容量(MW)'][:] * range + min
    print(data)
    return np.array(x)

# use random to reorder data
def shuffle(x, y):
    np.random.seed(int(time.time()))
    randomList = np.arange(x.shape[0])
    np.random.shuffle(randomList)
    return x[randomList], y[randomList]

# use LSTM model as training model
def buildManyToManyModel(shape, batch_size):
    model = Sequential()
    model.add(LSTM(64, input_shape=(shape[1], shape[2]), return_sequences = True))
    model.add(Dropout(0.2))
    # # Adding a second LSTM layer and some Dropout regularisation
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))
    
    # # Adding a third LSTM layer and some Dropout regularisation
    # model.add(LSTM(64, return_sequences = True))
    # model.add(Dropout(0.2))
    
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(TimeDistributed(Dense(units = 1)))
    model.add(Flatten())
    model.add(Dense(64, activation='linear'))
    # output 7-days prediction of 備轉容量(MW)
    model.add(Dense(7))
    model.compile(loss="mse", optimizer="adam", metrics=['mse'])
    model.summary()
    return model

# test 10 data and plot the prediction result, correct answer, and the abs between the two data
def predictTestDump(precict_y, ans_y):
    fig, ax = plt.subplots(2, 5, figsize = (25, 14))
    for i in range(10):
        ax[i//5, i%5].plot(precict_y[i], label='predict')
        ax[i//5, i%5].plot(ans_y[i], label='ans')
        ax[i//5, i%5].plot(np.abs(precict_y[i] - ans_y[i]), label='error')
        ax[i//5, i%5].set_xlabel('test data')
        ax[i//5, i%5].set_ylabel('res')
        ax[i//5, i%5].set_title('predict res')
        ax[i//5, i%5].legend()
    plt.savefig('predict.png')

# plot the loss, mse diagram and save it
def lossDump(history):
    fig, ax = plt.subplots()
    ax.plot(history['val_loss'], label='val_loss')
    ax.plot(history['val_mse'], label='val_mse')
    ax.plot(history['loss'], label='loss')
    ax.plot(history['mse'], label='mse')
    ax.set_ylabel('result')
    ax.set_xlabel('epoch')
    ax.set_title('history')
    ax.legend()
    plt.savefig('loss.png')



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--training',
                       default='training_data.csv',
                       help='input training data file name')

    parser.add_argument('--output',
                        default='submission.csv',
                        help='output file name')
    args = parser.parse_args()

    # important arguments
    batch_size = 64
    ref_day = 30
    predict_day = 7
    train_ratio = 0.85
    epoch = 500
    patience = 50

    # data loading and preprocessing
    train_data = readCSV(args.training)
    min_, max_, range_ = np.min(train_data['備轉容量(MW)']), np.max(train_data['備轉容量(MW)']), (np.max(train_data['備轉容量(MW)']) - np.min(train_data['備轉容量(MW)']))
    data_norm = normalize(train_data)
    x_train, x_test, y_train, y_test = setTrainTestData(data_norm, ref_day, predict_day, train_ratio)
    x_train, y_train = shuffle(x_train, y_train)
    x_test, y_test = shuffle(x_test, y_test)

    # build model, use EarlyStopping with validation mse monitor as callbacks
    model = buildManyToManyModel(x_train.shape, batch_size)
    early_stopping = EarlyStopping(monitor='val_mse', patience=patience, verbose=1, mode='min')
    history = model.fit(x_train, y_train, verbose=1, callbacks=[early_stopping],\
        validation_data=(x_test, y_test), batch_size=batch_size, epochs=epoch)

    lossDump(history.history)

    # predict 10 data and plot the prediction result to check whether the model was well trained
    x_dump = x_test[:10]
    y_dump = y_test[:10]
    y_predict_dump = model.predict(x_dump)
    y_dump = y_dump[:] * range_ + min_
    y_predict_dump = y_predict_dump[:] * range_ + min_
    predictTestDump(y_predict_dump, y_dump)

    # use the latest data to predict the follow 7 days' 備轉容量(MW)
    predict_target = setPredictInput(data_norm, ref_day, min_, range_)
    df_result = model.predict(predict_target)[0]
    df_result = np.round(df_result[:] * range_ + min_, decimals=2)

    # write prediction result to csv file
    predict_dict = {}
    predict_dict['date'] = ['20210323', '20210324', '20210325', '20210326', '20210327', '20210328', '20210329']
    predict_dict['operating_reserve(MW)'] = df_result
    df = pd.DataFrame(predict_dict, columns= ['date', 'operating_reserve(MW)'])
    df.to_csv (args.output, index = False, header=True)
    print (df)

    # save model
    save_name = '{}_{}.h5'.format(int(time.time()), np.around(np.min(history.history['val_mse']), decimals=4))
    model.save(save_name)
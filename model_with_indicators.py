import datetime as dt
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.metrics import accuracy_score

ticker = 'AAPL'
start = dt.datetime(2014,1,1)
end = dt.datetime(2019,4,25)
dl = web.DataReader(ticker,'yahoo', start, end)
dl.to_csv('C:/Users/Никита/Desktop/Python Programs/my model/'+ticker+'.csv')

def moving_average(df, n):

    MA = pd.Series(df['Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
    df = df.join(MA)
    return df

def macd(df, n_fast, n_slow):
    """Calculate MACD, MACD Signal and MACD difference
    """
    EMAfast = pd.Series(df['Close'].ewm(span=n_fast, min_periods=n_slow).mean())
    EMAslow = pd.Series(df['Close'].ewm(span=n_slow, min_periods=n_slow).mean())
    MACD = pd.Series(EMAfast - EMAslow, name='MACD_' + str(n_fast) + '_' + str(n_slow))
    MACDsign = pd.Series(MACD.ewm(span=9, min_periods=9).mean(), name='MACDsign_' + str(n_fast) + '_' + str(n_slow))
    MACDdiff = pd.Series(MACD - MACDsign, name='MACDdiff_' + str(n_fast) + '_' + str(n_slow))
    df = df.join(MACD)
    df = df.join(MACDsign)
    df = df.join(MACDdiff)
    return df	
	
def relative_strength_index(df, n):

    i = df.index[0]
    UpI = [0]
    DoI = [0]
    while i + 1 <= df.index[-1]:
        UpMove = float(df.loc[i + 1, 'High']) - float(df.loc[i, 'High'])
        DoMove = float(df.loc[i, 'Low']) - float(df.loc[i + 1, 'Low'])
        if UpMove > DoMove and UpMove > 0:
            UpD = UpMove
        else:
            UpD = 0
        UpI.append(UpD)
        if DoMove > UpMove and DoMove > 0:
            DoD = DoMove
        else:
            DoD = 0
        DoI.append(DoD)
        i = i + 1
    UpI = pd.Series(UpI)

    DoI = pd.Series(DoI)
    PosDI = pd.Series(UpI.ewm(span=n, min_periods=n).mean())
    NegDI = pd.Series(DoI.ewm(span=n, min_periods=n).mean())

    # rsi = pd.Series(PosDI / (PosDI + NegDI), name='RSI_' + str(n))
    rsi = pd.DataFrame(PosDI / (PosDI + NegDI), columns=['RSI_' + str(n)])
    rsi = rsi.set_index(df.index)
    df = df.join(rsi)
    return df	
	
def stochastic_oscillator_k(df):
    
    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO_k')
    df = df.join(SOk)
    return df

def stochastic_oscillator_d(df, n):

    SOk = pd.Series((df['Close'] - df['Low']) / (df['High'] - df['Low']), name='SO%k')
    SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SO_' + str(n))
    df = df.join(SOd)
    return df
	
def rate_of_change(df, n):

    M = df['Close'].diff(n - 1)
    N = df['Close'].shift(n - 1)
    ROC = pd.Series(M / N, name='ROC_' + str(n))
    df = df.join(ROC)
    return df	
	
def mass_index(df):

    Range = df['High'] - df['Low']
    EX1 = Range.ewm(span=9, min_periods=9).mean()
    EX2 = EX1.ewm(span=9, min_periods=9).mean()
    Mass = EX1 / EX2
    MassI = pd.Series(Mass.rolling(25).sum(), name='Mass Index')
    df = df.join(MassI)
    return df	
	
def bollinger_bands(df, n, std, add_ave=True):

    ave = df['Close'].rolling(window=n, center=False).mean()
    sd = df['Close'].rolling(window=n, center=False).std()
    upband = pd.Series(ave + (sd * std), name='bband_upper_' + str(n))
    dnband = pd.Series(ave - (sd * std), name='bband_lower_' + str(n))
    if add_ave:
        ave = pd.Series(ave, name='bband_ave_' + str(n))
        df = df.join(pd.concat([upband, dnband, ave], axis=1))
    else:
        df = df.join(pd.concat([upband, dnband], axis=1))

    return df

def commodity_channel_index(df, n):

    PP = (df['High'] + df['Low'] + df['Close']) / 3
    CCI = pd.Series((PP - PP.rolling(n, min_periods=n).mean()) / PP.rolling(n, min_periods=n).std(),
                    name='CCI_' + str(n))
    df = df.join(CCI)
    return df
	
def on_balance_volume(df, n):

    i = 0
    OBV = [0]
    while i < df.index[-1]:
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] > 0:
            OBV.append(df.loc[i + 1, 'Volume'])
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] == 0:
            OBV.append(0)
        if df.loc[i + 1, 'Close'] - df.loc[i, 'Close'] < 0:
            OBV.append(-df.loc[i + 1, 'Volume'])
        i = i + 1
    OBV = pd.Series(OBV)
    OBV_ma = pd.Series(OBV.rolling(n, min_periods=n).mean(), name='OBV_' + str(n))
    df = df.join(OBV_ma)
    return df	
	
data = pd.read_csv(''+ticker+'.csv')
data = data.drop(['Date','Open','Adj Close'], 1)

data = moving_average(df=data, n=12)
data = macd(df=data, n_fast=12, n_slow=26)
data = relative_strength_index(df=data, n=14)
data = stochastic_oscillator_k(df=data)
data = stochastic_oscillator_d(df=data, n=5)
data = rate_of_change(df=data, n=10)
data = mass_index(df=data)
data = bollinger_bands(df=data, n=20, std=4, add_ave=False)
data = commodity_channel_index(df=data, n=14)
data = on_balance_volume(df=data, n=12)

print(data.tail())

#data.to_csv('table.csv')

# Cut DataFrame
data = data.iloc[40::]
# Reset index
data = data.reset_index()
# Delete old index
data = data.drop('index', 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(-1, 1))
data_n = pd.DataFrame(scaler.fit_transform(data),columns=data.columns.values)

print(data_n.tail())

# 1) Prepare datasets

features = ['MA_12', 'MACD_12_26', 'MACDsign_12_26', 'MACDdiff_12_26', 'RSI_14', 'SO_k', 'SO_5', 'ROC_10', 'Mass Index', 'bband_upper_20', 'bband_lower_20', 'CCI_14', 'OBV_12']
test_dataset_size = 0.05
dataset_train_length = len(data_n.index) -\
    int(len(data_n.index) * test_dataset_size)

number_of_features = len(features)
training_data = data_n.iloc[:dataset_train_length]

trainX = data_n[features].values[:dataset_train_length]
trainY = data_n['Close'].values[:dataset_train_length]
testX = data_n[features].values[dataset_train_length:]
testY = data_n['Close'].values[dataset_train_length:]

# RESHAPING TRAIN AND TEST DATA
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# LSTM MODEL
model = Sequential()
model.add(LSTM(
    input_shape=(None, number_of_features),
    units=50,
    return_sequences=True))

model.add(LSTM(
    100,
    return_sequences=False))

model.add(Dense(
    units=1))
model.add(Activation('linear'))

# MODEL COMPILING AND TRAINING
model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['accuracy']) #  SGD, adam, adagrad 
history = model.fit(trainX, trainY, epochs=100, batch_size=32 ,verbose=1, validation_data=(testX, testY))

print(model.summary())

plt.rcParams.update({'font.size': 18})
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training error', 'Test error'], loc = 'upper right')
plt.show()

result1 = model.evaluate(trainX, trainY)
results = model.evaluate(testX, testY)

print('Loss: ', result1[0])
print('val_loss: ', results[0])

# PREDICTION
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# DE-NORMALIZING FOR PLOTTING
scaler = MinMaxScaler(feature_range=(-1, 1))

close = data[['Close']]
close = np.reshape(close.values, (len(close),1))
close = scaler.fit_transform(close)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
trainPredictPlot = np.empty_like(close)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[0:dataset_train_length, :] = trainPredict

# CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
testPredictPlot = np.empty_like(close)
testPredictPlot[:, :] = np.nan
testPredictPlot[dataset_train_length:len(data_n.index), :] = testPredict

dataset_train_length - 1, len(data_n.index)

# DE-NORMALIZING MAIN DATASET 
close = scaler.inverse_transform(close)

# PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
plt.plot(close, 'g', label = 'original dataset')
plt.plot(trainPredictPlot, 'r', label = 'training set')
plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
plt.legend(loc = 'upper left')
plt.xlabel('Time in Days')
plt.ylabel('close Value of '+ticker+'')
plt.show()
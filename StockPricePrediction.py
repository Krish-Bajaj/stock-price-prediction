# IMPORTING IMPORTANT LIBRARIES
from cProfile import label
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tweepy
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from sklearn import datasets
import numpy as np
from yahoo_historical import Fetcher
from tweet import gettweetdata


# FUNCTION TO CREATE 1D DATA INTO TIME SERIES DATASET
def new_dataset(dataset, step_size):
    data_X, data_Y = [],[]
    for i in range(len(dataset) - step_size - 1):
            a = dataset[i:(i + step_size), 0]
            data_X.append(a)
            data_Y.append(dataset[i + step_size, 0])
    return np.array(data_X), np.array(data_Y)


# FOR REPRODUCIBILITY
def get_predicted_data(symbol):

    np.random.seed(7)

    # Fetching historical data
    company=symbol
    data = Fetcher(company, [2021, 4, 30], [2022, 4, 30])      #**************
    df = pd.DataFrame(data.getHistorical())
    df.dropna()
    #df=df.iloc[::-1]
    df.to_csv(r'./stock.csv')

    # Get sentiment analysis data
    sentiment = gettweetdata(('$'+company))

    # IMPORTING DATASET 
    dataset = pd.read_csv('./stock.csv', usecols=[2,3,4,5])
    #dataset = dataset.reindex(index = dataset.index[::-1])

    # CREATING OWN INDEX FOR FLEXIBILITY
    obs = np.arange(1, len(dataset) + 1, 1)

    # TAKING DIFFERENT INDICATORS FOR PREDICTION
    OHLC_avg = dataset.mean(axis = 1)
    HLC_avg = dataset[['High', 'Low', 'Close']].mean(axis = 1)
    close_val = dataset[['Close']]

    # PLOTTING ALL INDICATORS IN ONE PLOT
    #plt.plot(obs, OHLC_avg, 'r', label = 'OHLC avg')
    #plt.plot(obs, HLC_avg, 'b', label = 'HLC avg')
    #plt.plot(obs, close_val, 'g', label = 'Closing price')
    #plt.legend(loc = 'upper right')
    #plt.show()

    # PREPARATION OF TIME SERIES DATASE
    OHLC_avg = np.reshape(OHLC_avg.values, (len(OHLC_avg),1)) # 1664
    scaler = MinMaxScaler(feature_range=(0, 1))
    OHLC_avg = scaler.fit_transform(OHLC_avg)

    # TRAIN-TEST SPLIT
    train_OHLC = int(len(OHLC_avg) * 0.9)
    test_OHLC = len(OHLC_avg) - train_OHLC
    train_OHLC, test_OHLC = OHLC_avg[0:train_OHLC,:], OHLC_avg[train_OHLC:len(OHLC_avg),:]

    # TIME-SERIES DATASET (FOR TIME T, VALUES FOR TIME T+1)
    trainX, trainY = new_dataset(train_OHLC, 1)
    testX, testY = new_dataset(test_OHLC, 1)


    # RESHAPING TRAIN AND TEST DATA
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    step_size = 1

    # LSTM MODEL
    model = Sequential()
    model.add(LSTM(100, input_shape=(1, step_size), return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(100,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Activation('linear'))

    # MODEL COMPILING AND TRAINING
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) # Try SGD, adam, adagrad and compare!!!
    model.fit(trainX, trainY, epochs=100, batch_size=32,verbose=1)

    # PREDICTION
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)


    # DE-NORMALIZING FOR PLOTTING
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])


    # TRAINING RMSE
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train RMSE: %.2f' % (trainScore))

    # TEST RMSE
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test RMSE: %.2f' % (testScore))

    # CREATING SIMILAR DATASET TO PLOT TRAINING PREDICTIONS
    trainPredictPlot = np.empty_like(OHLC_avg)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[step_size:len(trainPredict)+step_size, :] = trainPredict

    # CREATING SIMILAR DATASSET TO PLOT TEST PREDICTIONS
    testPredictPlot = np.empty_like(OHLC_avg)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(step_size*2)+1:len(OHLC_avg)-1, :] = testPredict


    # DE-NORMALIZING MAIN DATASET 
    OHLC_avg = scaler.inverse_transform(OHLC_avg)

    # # PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
    # last_val = "1"
    # nextdata = "2"
    # accuracy = "3"
    # # text = 'Last day value: ' + last_val + "\nNext day value: " + nextdata + "\nAccuracy: " + accuracy
    # plt.plot(OHLC_avg, 'g', label = 'original dataset')
    # plt.plot(trainPredictPlot, 'r', label = 'training set')
    # plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
    # plt.legend(loc = 'upper right')
    # plt.xlabel('Time in Days')
    # plt.ylabel('Value of '+company +' Stock')
    # # plt.title(text)
    # # plt.show()


    predicted=np.append(trainPredict,testPredict)
    pdf=pd.DataFrame(predicted)
    #df=df.drop([0,1,2,3])
    df.insert(7, 'Predicted', pdf)
    df.to_csv(r'./pstock.csv')


    # PREDICT FUTURE VALUES
    last_val = testPredict[-1]
    last_val_scaled = last_val/last_val
    next_val = model.predict(np.reshape(last_val_scaled, (1,1,1)))
    print("Last Day Value:", np.ndarray.item(last_val))
    nextdata=np.ndarray.item(last_val*next_val)
    nextdata=nextdata+(sentiment*sentiment/nextdata)
    print("Next Day Value:", nextdata)
    print(np.append(last_val, next_val))
    accuracy = 0
    if nextdata > np.ndarray.item(last_val):
        accuracy = 100 - ((nextdata - np.ndarray.item(last_val))/nextdata)*100
    else:
        accuracy = 100 - ((np.ndarray.item(last_val) - nextdata)/nextdata)*100

    # PLOT OF MAIN OHLC VALUES, TRAIN PREDICTIONS AND TEST PREDICTIONS
    font = {'family': 'serif',
        'size': 10,
        }
    last_val1 = "{:.2f}".format(np.ndarray.item(last_val))
    nextdata1 = "{:.2f}".format(nextdata)
    accuracy1 = "{:.2f}".format(accuracy)
    # text = 'Last day value: ' + last_val + "\nNext day value: " + nextdata + "\nAccuracy: " + accuracy
    plt.plot(OHLC_avg, 'g', label = 'original dataset')
    plt.plot(trainPredictPlot, 'r', label = 'training set')
    plt.plot(testPredictPlot, 'b', label = 'predicted stock price/test set')
    plt.legend(loc = 'upper right')
    plt.xlabel('Time in Days')
    plt.ylabel('Value of '+company +' Stock')
    # plt.title(text)
    # plt.show()
    text = 'Last day value: $' + last_val1 + "\nNext day value: $" + nextdata1 + "\nAccuracy: " + accuracy1 + "%"
    plt.title(text, fontdict=font)
    plt.show()
    return np.ndarray.item(last_val), nextdata, accuracy
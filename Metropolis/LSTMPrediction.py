# LSTM for international airline passengers problem with window regression framing
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# convert an array of values into a dataset matrix


class LSTMPrediction:
    def __init__(self, data, split_ratio, numofLSTMLayers=4, batch_size=1, epochs=100, look_back=3, numOfFeatures = 1):
        # fix random seed for reproducibility
        numpy.random.seed(7)
        self.look_back = look_back
        self.split_ratio = split_ratio
        self.load_data(data)
        self.LSTM_layers = numofLSTMLayers
        self.batch_size = batch_size
        self.epochs = epochs
        self.numOfFeatures = numOfFeatures

        self.model = Sequential()
        self.model.add(LSTM(numofLSTMLayers, input_shape=(self.numOfFeatures, self.look_back)))
        self.model.add(Dense(self.numOfFeatures))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-look_back-1):
            a = dataset[i:(i+look_back), :]
            dataX.append(a)
            dataY.append(dataset[i + look_back, :])
        return numpy.array(dataX), numpy.array(dataY)

    def load_data(self, data):
        #ie data = 'airline-passengers.csv'
        if type(data) == str:
            self.dataframe = read_csv(data, usecols=[1], engine='python')
            self.dataset = self.dataframe.values
            self.dataset = self.dataset.astype('float32')
        else:
            self.dataset = data

    def transform_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.dataset = self.scaler.fit_transform(self.dataset)

    def create_test_and_train_data(self, split_ratio=0.67):
        self.transform_data()
        # split into train and test sets
        train_size = int(len(self.dataset) * split_ratio)
        test_size = len(self.dataset) - train_size
        train, test = self.dataset[0:train_size, :], self.dataset[train_size:len(self.dataset), :]
        # reshape into X=t and Y=t+1

        self.trainX, self.trainY = self.create_dataset(train, self.look_back)
        self.testX, self.testY = self.create_dataset(test, self.look_back)
        # reshape input to be [samples, time steps, features]

        self.trainX = numpy.reshape(self.trainX, (self.trainX.shape[0], self.trainX.shape[2], self.trainX.shape[1]))
        self.testX = numpy.reshape(self.testX, (self.testX.shape[0], self.testX.shape[2], self.testX.shape[1]))
        # try:
        #     self.testX = numpy.reshape(self.testX, (self.testX.shape[0], 1, self.testX.shape[1]))
        # except:
        #     print("hiba")

    def train(self, split_ratio):
        self.create_test_and_train_data(split_ratio)
        self.model.fit(self.trainX, self.trainY, epochs=self.epochs, batch_size=self.batch_size, verbose=2)

    def prediction(self, split_ratio):
        # create and fit the LSTM network
        self.train(split_ratio)
        # make predictions
        trainPredict = self.model.predict(self.trainX)
        testPredict = self.model.predict(self.testX)
        testPredict2 = self.model.predict(self.testX)
        # test_data = self.testX[0].reshape((1, 1, self.look_back)) #self.numOfFeatures
        # for i in range(len(self.testY)):
        #     # print(self.testX[i])
        #     # print(test_data)
        #     testPredict2[i, 0] = self.model.predict(test_data)
        #     # print(self.model.predict(self.testX)[i,0])
        #     # print(self.testY[i])
        #     # print(testPredict2[i,0])
        #     test_data = numpy.roll(test_data, -1)
        #     test_data[0, 0, -1] = testPredict2[i, 0]
        #     # for j in range(self.look_back-1):
        #     #     test_data[0,0,j] = self.testX[i,0,j+1]
        #     # test_data[0, 0, 8] = self.testX[i, 0, 9]

        # invert predictions
        trainPredict = self.scaler.inverse_transform(trainPredict)
        self.trainY = self.scaler.inverse_transform(self.trainY)#self.trainY = self.scaler.inverse_transform([self.trainY])
        testPredict = self.scaler.inverse_transform(testPredict)
        testPredict2 = self.scaler.inverse_transform(testPredict2)
        self.testY = self.scaler.inverse_transform(self.testY)#self.testY = self.scaler.inverse_transform([self.testY])
        # calculate root mean squared error
        # trainScore = math.sqrt(mean_squared_error(self.trainY[:], trainPredict[:,0]))
        # print('Train Score: %.2f RMSE' % (trainScore))
        # testScore = math.sqrt(mean_squared_error(self.testY[:], testPredict[:,0]))
        # print('Test Score: %.2f RMSE' % (testScore))
        # shift train predictions for plotting
        trainPredictPlot = numpy.empty_like(self.dataset)
        trainPredictPlot[:, :] = numpy.nan
        trainPredictPlot[self.look_back:len(trainPredict)+self.look_back, :] = trainPredict
        trainPredictPlot = trainPredictPlot[~numpy.isnan(trainPredictPlot)]
        # shift test predictions for plotting
        testPredictPlot = numpy.empty_like(self.dataset)
        testPredictPlot[:, :] = numpy.nan
        testPredictPlot[len(trainPredict)+(self.look_back*2)+1:len(self.dataset)-1, :] = testPredict
        testPredictPlot = testPredictPlot[~numpy.isnan(testPredictPlot)]
        testPredictPlot2 = numpy.empty_like(self.dataset)
        testPredictPlot2[:, :] = numpy.nan
        testPredictPlot2[len(trainPredict)+(self.look_back*2)+1:len(self.dataset)-1, :] = testPredict2
        testPredictPlot2 = testPredictPlot2[~numpy.isnan(testPredictPlot2)]
        # return trainPredictPlot, testPredictPlot, testPredictPlot2
        return trainPredict, testPredict, testPredict2
        # plot baseline and predictions
        # plt.plot(self.scaler.inverse_transform(self.dataset))
        # plt.plot(trainPredictPlot)
        # plt.plot(testPredictPlot)
        # plt.show()

    def predictNextStep(self, inputData):
        self.dataset = inputData
        self.transform_data()
        input = []
        for i in range(len(inputData)-self.look_back + 1):
            a = self.dataset[i : (i + self.look_back), 0]
            input.append(a)
        # input, output = self.create_dataset(self.dataset, self.look_back)
        input = numpy.array(input)
        input = numpy.reshape(input, (input.shape[0], 1, input.shape[1]))
        prediction = self.model.predict(input)
        prediction = self.scaler.inverse_transform(prediction)

        return prediction
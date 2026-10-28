import numpy as np
from sklearn.model_selection import train_test_split


def input_to_hidden(x, Win):
    a = np.dot(x, Win)
    return np.maximum(a, 0, a)
    #
    # a = np.maximum(a, 0, a) # ReLU
    # return a

def predict(x, Win, Wout):
    return np.dot(input_to_hidden(x, Win), Wout)
    # x = input_to_hidden(x, Win)
    # y = np.dot(x, Wout)
    # return y

def classification(x_train, y_train, x_test, y_test, hiddenUnits=30):

    INPUT_LENGHT = x_train.shape[1]
    HIDDEN_UNITS = hiddenUnits

    Win = np.random.normal(size=[INPUT_LENGHT, HIDDEN_UNITS])
    print('Input Weight shape: {shape}'.format(shape=Win.shape))

    X = input_to_hidden(x_train,Win)
    Xt = np.transpose(X)
    Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, y_train))
    print('Output weights shape: {shape}'.format(shape=Wout.shape))


    y = predict(x_test,Win,Wout)
    correct = 0
    total = y.shape[0]
    for i in range(total):
        predicted = np.argmax(y[i])
        test = np.argmax(y_test[i])
        correct = correct + (1 if predicted == test else 0)
    print('Accuracy: {:f}'.format(correct/total))

    return Win, Wout
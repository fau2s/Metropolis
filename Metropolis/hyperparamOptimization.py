from functools import partial

import numpy as np
import optuna
from keras.callbacks import EarlyStopping
from keras.layers import Input, LSTM, BatchNormalization, RepeatVector, TimeDistributed, Dense
from keras.models import Model
from tensorflow.keras.optimizers import Adam
import LuongAttention
from NBEATS import NBEATS
from NHiTS import NHITS


def truncate(x, feature_cols=range(3), target_cols=range(3), label_col=3, train_len=130, test_len=20):
    in_, out_, lbl = [], [], []
    for i in range(len(x) - train_len - test_len + 1):
        in_.append(x[i:(i + train_len), feature_cols].tolist())
        out_.append(x[(i + train_len):(i + train_len + test_len), target_cols].tolist())
        lbl.append(x[i + train_len, label_col])
    return np.array(in_), np.array(out_), np.array(lbl)


def optimize(params, x, y):

    # n_hidden = 10 #param to optimize
    # trunc_train_len = 150
    # trunc_test_len = 10
    # momentum = 0.1 #param to optimize
    # learning_rate = 0.015  # param to optimize
    # epc = 800  # param to optimize
    # batch_size = 50  # param to optimize
    n_hidden = params['n_hidden'] #param to optimize
    # trunc_train_len = params['trunc_train_len']
    # trunc_test_len = params['trunc_train_len']
    momentum = params['momentum'] #param to optimize
    learning_rate = params['learning_rate']  # param to optimize
    epc = params['epc']  # param to optimize
    batch_size = params['batch_size']  # param to optimize
    input_train = Input(shape=(x.shape[1], x.shape[2] - 1))
    output_train = Input(shape=(y.shape[1], y.shape[2] - 1))
    encoder_last_h1, encoder_last_h2, encoder_last_c = LSTM(
        units=n_hidden,
        activation='tanh',
        dropout=0.2,
        recurrent_dropout=0.2,
        return_sequences=False,
        return_state=True
    )(input_train)

    encoder_last_h1 = BatchNormalization(momentum=momentum)(encoder_last_h1)
    encoder_last_c = BatchNormalization(momentum=momentum)(encoder_last_c)

    decoder = RepeatVector(output_train.shape[1])(encoder_last_h1)
    decoder = LSTM(
        units=n_hidden,
        activation='tanh',
        dropout=0.2,
        recurrent_dropout=0.2,
        return_state=False,
        return_sequences=True
    )(decoder, initial_state=[encoder_last_h1, encoder_last_c])
    out = TimeDistributed(Dense(output_train.shape[2]))(decoder)

    model = Model(inputs=input_train,
                       outputs=out)
    opt = Adam(learning_rate=learning_rate,
                    clipnorm=1)
    model.compile(loss='mean_squared_error',
                       optimizer=opt,
                       metrics=['mae'])

    # model = ensemble.RandomForestClassifier(**params)
    # kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []
    # for idx in kf.split(X=x, y=y):
        # train_idx, test_idx = idx[0], idx[1]
        # xtrain = x[train_idx]
        # ytrain = y[train_idx]
        #
        # xtest = x[test_idx]
        # ytest = y[test_idx]

    es = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=50
    )
    history = model.fit(
        X_input_train[:, :, :1],
        X_output_train[:, :, :1],
        validation_split=0.2,
        epochs=epc,
        verbose=0,
        callbacks=[es],
        batch_size=batch_size
    )
    train_mae = history.history['mae']
    valid_mae = history.history['val_mae']
    train_pred_detrend = model.predict(X_input_train[:, :, :1])#*x_train_max[:1]
    test_pred_detrend = model.predict(X_input_test[:, :, :1])#*x_train_max[:1]
    # model.fit(xtrain, ytrain)
    # preds = model.predict(train_pred_detrend)
    # metrics.accuracy_score(test_pred_detrend.flatten().tolist(),test_pred_detrend.flatten().tolist())
    # fold_acc = metrics.accuracy_score(test_pred_detrend.flatten(), test_pred_detrend.flatten())
    accuracies.append(valid_mae)

    return np.mean(accuracies)


def optimize_optuna(trial, testdata, truncation_test_len, ratio):

    trunc_test_len = truncation_test_len
    # no attention with tanh activation
    trunc_train_len = trial.suggest_int("trunc_train_len", 50, 250, step=20)
    n_hidden = trial.suggest_categorical("n_hidden", [64, 128, 256, 512])  #n_hidden = trial.suggest_int("n_hidden", 80, 300,)
    momentum = trial.suggest_float("momentum", 0.05, 0.85, step=0.2) #trial.suggest_uniform("momentum", 0.4, 0.95)
    learning_rate = trial.suggest_categorical("learning_rate", [0.000001, 0.00001, 0.00005, 0.0001, 0.0005]) #trial.suggest_uniform("learning_rate", 0.02, 0.195)
    epc = trial.suggest_int("epc", 250, 350, step=50)
    batch_size = trial.suggest_int("batch_size", 20, 140, step=20)

    dataLength = len(testdata)

    accuracies = []
    accuracies_shifted = []
    nlpLSTMPred = LuongAttention.NLPwithLSTM(
        data=testdata,
        train_ratio=ratio,
        n_hidden=n_hidden,
        momentum=momentum,
        epc=epc,
        learning_rate=learning_rate,
        batch_size=batch_size,
        train_len=trunc_train_len,
        test_len=trunc_test_len,
        attention="Luong"
    )
    nlpLSTMPred.constructnetwork()
    nlpLSTMPred.trainnetwork()

    train_mae = nlpLSTMPred.train_mae
    valid_mae = nlpLSTMPred.valid_mae
    valid_mae_shifted = nlpLSTMPred.valid_mae_shifted
    accuracies.append(valid_mae)
    accuracies_shifted.append(valid_mae_shifted)
    nlpLSTMPred.plotPredict(
        save=True,
        spar=11,
        start=1,
        length=dataLength,
        inputLen=trunc_train_len,
        layers=2,
        nHidden=n_hidden,
        error=[np.mean(accuracies), np.mean(accuracies_shifted)],
        train_len=trunc_train_len
    )

    return np.mean(accuracies) #np.mean(accuracies_shifted)#np.mean(accuracies) #(np.mean(accuracies), nlpLSTMPred)


def optimize_optuna_NBEATS(trial, testdata, truncation_test_len, ratio):
    trunc_test_len = truncation_test_len
    # no attention with tanh activation
    trunc_train_len = trial.suggest_int("trunc_train_len", 50, 250, step=20)
    n_hidden = trial.suggest_categorical("n_hidden",
                                         [64, 128, 256, 512])  # n_hidden = trial.suggest_int("n_hidden", 80, 300,)
    n_stacks = trial.suggest_categorical("n_stacks",
                                         [3, 5, 8, 12])
    n_blocks = trial.suggest_categorical("n_blocks",
                                         [3, 5, 8, 12])
    n_layers = trial.suggest_categorical("n_layers",
                                         [3, 5, 8, 12])
    learning_rate = trial.suggest_categorical("learning_rate", [0.000001, 0.00001, 0.00005, 0.0001,
                                                                0.0005])  # trial.suggest_uniform("learning_rate", 0.02, 0.195)
    epc = trial.suggest_int("epc", 250, 350, step=50)
    batch_size = trial.suggest_int("batch_size", 20, 140, step=20)

    dataLength = len(testdata)

    accuracies = []
    accuracies_shifted = []

    NBeatsPred = NBEATS(
        data=testdata,
        train_ratio=ratio,
        epc=epc,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_stacks=n_stacks,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_hidden=n_hidden,
        train_len=trunc_train_len,
        test_len=trunc_test_len,
    )
    NBeatsPred.constructnetwork()
    NBeatsPred.trainnetwork()

    train_mae = NBeatsPred.train_mae
    valid_mae = NBeatsPred.valid_mae
    valid_mae_shifted = NBeatsPred.valid_mae_shifted
    accuracies.append(valid_mae)
    accuracies_shifted.append(valid_mae_shifted)
    NBeatsPred.plotPredict(
        save=True,
        spar=11,
        start=1,
        length=dataLength,
        inputLen=trunc_train_len,
        layers=n_layers,
        nHidden=n_hidden,
        error=[np.mean(accuracies), np.mean(accuracies_shifted)],
        train_len=trunc_train_len
    )

    return np.mean(accuracies)  # np.mean(accuracies_shifted)#np.mean(accuracies) #(np.mean(accuracies), nlpLSTMPred)


def optimize_optuna_NHITS(trial, testdata, truncation_test_len, ratio):
    trunc_test_len = truncation_test_len
    # no attention with tanh activation
    trunc_train_len = trial.suggest_int("trunc_train_len", 50, 250, step=20)
    n_hidden = trial.suggest_categorical("n_hidden",
                                         [64, 128, 256, 512])  # n_hidden = trial.suggest_int("n_hidden", 80, 300,)
    learning_rate = trial.suggest_categorical("learning_rate", [0.000001, 0.00001, 0.00005, 0.0001,
                                                                0.0005])  # trial.suggest_uniform("learning_rate", 0.02, 0.195)
    epc = trial.suggest_int("epc", 250, 350, step=50)
    batch_size = trial.suggest_int("batch_size", 20, 140, step=20)
    n_stacks = trial.suggest_categorical("n_stacks",
                                         [3, 5, 8])
    n_blocks = trial.suggest_categorical("n_blocks",
                                         [3, 5, 8])
    n_layers = trial.suggest_categorical("n_layers",
                                         [3, 5, 8])
    dataLength = len(testdata)

    accuracies = []
    accuracies_shifted = []
    nHitsPred = NHITS(
        data=testdata,
        train_ratio=ratio,
        epc=epc,
        batch_size=batch_size,
        learning_rate=learning_rate,
        n_stacks=n_stacks,
        n_blocks=n_blocks,
        n_layers=n_layers,
        n_hidden=n_hidden,
        train_len=trunc_train_len,
        test_len=trunc_test_len
    )
    nHitsPred.constructnetwork()
    nHitsPred.trainnetwork()

    train_mae = nHitsPred.train_mae
    valid_mae = nHitsPred.valid_mae
    valid_mae_shifted = nHitsPred.valid_mae_shifted
    accuracies.append(valid_mae)
    accuracies_shifted.append(valid_mae_shifted)
    nHitsPred.plotPredict(
        save=True,
        spar=11,
        start=1,
        length=dataLength,
        inputLen=trunc_train_len,
        layers=n_layers,
        nHidden=n_hidden,
        error=[np.mean(accuracies), np.mean(accuracies_shifted)],
        train_len=trunc_train_len,
        stacks=n_stacks,
        blocks=n_blocks
    )

    return np.mean(accuracies)  # np.mean(accuracies_shifted)#np.mean(accuracies) #(np.mean(accuracies), nlpLSTMPred)

if __name__ == "__main__":
    testdata = np.loadtxt('testdata4NLP.txt')
    # df = pd.read_csv("../input/mobile_train.csv")

    # trunc_train_len = 150
    trunc_test_len = 20
    #
    # train_ratio = 0.8
    # train_len = int(train_ratio * len(testdata))
    # test_len = int((1 - train_ratio) * len(testdata))
    # x_index = np.array(range(len(testdata)))
    #
    # x_lbl = np.column_stack(
    #     [testdata, x_index, [1] * train_len + [0] * (len(x_index) - train_len)]
    # )
    #
    # X_in, X_out, lbl = truncate(
    #     x_lbl,
    #     feature_cols=range(2),
    #     target_cols=range(2),
    #     label_col=2,
    #     train_len=trunc_train_len,
    #     test_len=trunc_test_len
    # )
    #
    # X_input_train = X_in[np.where(lbl == 1)]
    # X_output_train = X_out[np.where(lbl == 1)]
    # X_input_test = X_in[np.where(lbl == 0)]
    # X_output_test = X_out[np.where(lbl == 0)]

    # X = df.drop("price_range", axis=1).values
    # y = df.price_range.values

    # param_space = {
    #     "n_hidden": scope.int(hp.quniform("n_hidden", 10, 100, 1)),
    #     "momentum": hp.uniform("momentum", 0.01, 1),
    #     "learning_rate": hp.uniform("learning_rate", 0.001, 2),
    #     "epc": scope.int(hp.quniform("epc", 1, 50, 1)),
    #     "batch_size": scope.int(hp.quniform("batch_size", 1, 150, 1)),
    # }

    optimization_function = partial(optimize_optuna, testdata=testdata, truncation_test_len=trunc_test_len)
    study = optuna.create_study(direction="minimize")
    study.optimize(optimization_function, n_trials=10)
    study.best_params.values()

    # trials = Trials()
    #
    # result = fmin(
    #     fn=optimization_function,
    #     space=param_space,
    #     algo=tpe.suggest,
    #     max_evals=100,
    #     trials=trials
    # )

    # print(result)

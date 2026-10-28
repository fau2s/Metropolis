import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
# class NHITS:
#
#     def __init__(self,
#                  input_length,
#                  forecast_length,
#                  n_features,
#                  n_blocks=3,
#                  hidden_units=256):
#
#         self.input_length = input_length
#         self.forecast_length = forecast_length
#         self.n_features = n_features
#         self.n_blocks = n_blocks
#         self.hidden_units = hidden_units
#
#         self.build_model()
#
#     def hits_block(self, x, forecast_size):
#
#         h = Dense(self.hidden_units, activation="relu")(x)
#         h = Dense(self.hidden_units, activation="relu")(h)
#
#         forecast = Dense(forecast_size)(h)
#
#         return forecast
#
#     def build_model(self):
#
#         inputs = Input(shape=(self.input_length, self.n_features))
#         x = Flatten()(inputs)
#
#         forecasts = []
#
#         scales = [4,2,1]
#
#         for s in scales:
#
#             size = self.forecast_length // s
#
#             block_forecast = self.hits_block(x, size)
#
#             block_forecast = Reshape((size,1))(block_forecast)
#
#             block_forecast = UpSampling1D(size=s)(block_forecast)
#
#             forecasts.append(block_forecast)
#
#         forecast = Add()(forecasts)
#
#         self.model = Model(inputs, forecast)
#
#         self.model.compile(
#             optimizer=tf.keras.optimizers.Adam(1e-3),
#             loss="mse",
#             metrics=["mae"]
#         )

class NHITS():

    def __init__(self, data,
                 train_ratio=0.8,
                 epc=100,
                 batch_size=64,
                 learning_rate=0.001,
                 n_stacks=3,
                 n_blocks=3,
                 n_layers=4,
                 n_hidden=256,
                 train_len=150,
                 test_len=30):

        self.data = data
        self.train_ratio = train_ratio

        self.train_len = int(train_ratio * len(data))
        self.test_len = int((1-train_ratio) * len(data))

        self.truncation_train_len = train_len
        self.truncation_test_len = test_len

        self.batch_size = batch_size
        self.epc = epc
        self.learning_rate = learning_rate

        self.n_stacks = n_stacks
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_hidden = n_hidden

        self.prepare_data()

    def prepare_data(self):

        x = self.data.astype(float)

        if x.ndim == 1:
            mean = x.mean()
            std = x.std()
            std = std if std > 1e-8 else 1.0
        else:
            mean = x.mean(axis=0, keepdims=True)
            std = x.std(axis=0, keepdims=True)
            std[std < 1e-8] = 1.0

        x = (x - mean) / std

        self.numberOfFeatures = 1 if len(self.data.shape) == 1 else self.data.shape[1]

        self.x_index = np.linspace(0, 1, len(self.data), dtype=np.float32)

        self.x_lbl = np.column_stack([
            x,
            self.x_index,
            [1] * self.train_len + [0] * (len(x) - self.train_len)
        ])

        X_in, X_out, lbl = self.truncate(
            self.x_lbl,
            feature_cols=range(self.numberOfFeatures + 1),
            target_cols=range(self.numberOfFeatures + 1),
            label_col=self.numberOfFeatures + 1,
            train_len=self.truncation_train_len,
            test_len=self.truncation_test_len
        )

        self.X_input_train = X_in[lbl == 1]
        self.X_output_train = X_out[lbl == 1]

        self.X_input_test = X_in[lbl == 0]
        self.X_output_test = X_out[lbl == 0]

    def truncate(self, x, feature_cols=range(3), target_cols=range(3), label_col=3, train_len=130, test_len=20):

        in_, out_, lbl = [], [], []

        for i in range(len(x)-train_len-test_len+1):

            in_.append(x[i:(i+train_len), feature_cols].tolist())
            out_.append(x[(i+train_len):(i+train_len+test_len), target_cols].tolist())
            lbl.append(x[i+train_len, label_col])

        return np.array(in_), np.array(out_), np.array(lbl)

    def nhits_block(self, x, pool_size):

        pooled = AveragePooling1D(pool_size=pool_size)(x)

        flat = Flatten()(pooled)

        h = flat

        for _ in range(self.n_layers):
            h = Dense(self.n_hidden, activation="relu")(h)

        theta = Dense(self.truncation_test_len)(h)

        forecast = Reshape((self.truncation_test_len, 1))(theta)

        return forecast

    def constructnetwork(self):

        input_layer = Input(shape=(self.truncation_train_len, 1))

        x = input_layer

        forecast = None

        pool_sizes = [8, 4, 2]

        for s in range(self.n_stacks):

            for b in range(self.n_blocks):

                pool = pool_sizes[min(s, len(pool_sizes)-1)]

                block_forecast = self.nhits_block(x, pool)

                if forecast is None:
                    forecast = block_forecast
                else:
                    forecast = Add()([forecast, block_forecast])

        self.model = Model(input_layer, forecast)

        opt = Adam(
            learning_rate=self.learning_rate,
            clipnorm=1
        )

        self.model.compile(
            loss="mse",
            optimizer=opt,
            metrics=["mae"]
        )


    def trainnetwork(self):

        es = EarlyStopping(
            monitor='val_loss',
            patience=30
        )

        self.model.fit(

            self.X_input_train[:, :, 0:1],
            self.X_output_train[:, :, 0:1],

            validation_split=0.2,
            epochs=self.epc,
            batch_size=self.batch_size,
            verbose=0,
            callbacks=[es]
        )
        self.train_mae = (
            abs(self.predict(self.X_input_train[:, :, :1])
                - self.X_output_train[:, :, :1]).mean(axis=(1, 2)))
        self.valid_mae = (
            abs(self.predict(self.X_input_test[:, :, :1]) -
                self.X_output_test[:, :, :1]).mean(axis=(1, 2)))

        size = self.predict(self.X_input_test[:, :, :1]).shape[0]
        a = (np.array([self.X_output_train[-1, :, 0] -
                       self.predict(self.X_input_test[:, :, :1])[0, :, 0] ] * size)
             )[:, :, np.newaxis]
        self.train_mae_shifted = (
            abs(self.predict(self.X_input_train[:, :, :1]) -
                self.X_output_train[:, :, :1]).mean(axis=(1, 2)))

        self.valid_mae_shifted = (
            abs(self.predict(self.X_input_test[:, :, :1]) + a -
                self.X_output_test[:, :, :1]).mean(axis=(1, 2)))

    def predict(self, X):

        return self.model.predict(X[:, :, 0:1], verbose=0)


    def plotPredict(self, save=False, spar=1, start=1, length=1, inputLen=20, layers=3, nHidden=100, error=0, train_len = 230, stacks=4, blocks=4, figure=None):
        train_pred_detrend = self.predict(self.X_input_train[:, :, :1])#*self.x_train_max[0]/10
        test_pred_detrend = self.predict(self.X_input_test[:, :, :1])#*self.x_train_max[0]/10
        train_true_detrend = self.X_output_train[:, :, :1]#*self.x_train_max[0]
        test_true_detrend = self.X_output_test[:, :, :1]#*self.x_train_max[0]

        data_final = dict()
        for dt, lb in zip([train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
                          ['train_pred', 'train_true', 'test_pred', 'test_true']):
            dt_x1 = dt[:, :, 0]
            dt_x2 = dt[:, :, 0] #/ self.x_train_max[0]
            data_final[lb] = np.concatenate([np.expand_dims(dt_x1, axis=2), np.expand_dims(dt_x2, axis=2)], axis=2)

        plt.figure(figsize=(15, 5))
        train_start_t = 0
        test_start_t = data_final['train_pred'].shape[1]
        x_lbl = 'x1'

        shift = (
                self.X_input_train[-1, -1, 0] -
                data_final['test_pred'][-1, 0, 0]
        )

        shift_2 = (
                self.X_input_train[-1, -1, 0] -
                data_final['train_pred'][-1, -1, 0]
        )


        pred = data_final['test_pred'][:, :, 0] + shift

        n_windows, horizon = pred.shape
        T = n_windows + horizon - 1

        sum_pred = np.zeros(T)
        count = np.zeros(T)

        for h in range(horizon):
            sum_pred[h:h + n_windows] += pred[:, h]
            count[h:h + n_windows] += 1

        avg_pred = sum_pred / count


        plt.plot(
            range(data_final['train_pred'].shape[1]),
            data_final['train_pred'][-1, :, 0] + shift,
            linestyle='--', linewidth=1, color='green', label='pred ' + x_lbl
        )

        plt.plot(
            range(data_final['train_pred'].shape[1]),
            data_final['train_true'][-1, :, 0],
            linestyle='-', linewidth=1, color='blue', label='true ' + x_lbl
        )

        plt.plot(
            range(test_start_t, test_start_t + data_final['test_pred'].shape[1]),
            data_final['test_pred'][-1, :, 0] + shift,
            linestyle='--', linewidth=1, color='red', label='pred ' + x_lbl
        )


        plt.plot(
            range(test_start_t, test_start_t + data_final['test_pred'].shape[1]),
            data_final['test_pred'][-1, :, 0] + shift_2,
            linestyle='-.', linewidth=1, color='red', label='pred ' + x_lbl
        )

        plt.plot(
            range(test_start_t, test_start_t + data_final['test_true'].shape[1]),
            data_final['test_true'][-1, :, 0],
            linestyle='-', linewidth=1, color='orange', label='true ' + x_lbl
        )


        plt.title('{}th time step in all samples'.format(0))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=8)
        if save:
            plt.savefig('NLPtestFigures_spar{spar}_start{start}_length{length}_stacks{stacks}_blocks{blocks}_layers{layers}_hidden{nHidden}_lr{lr}_train_len{train_len}_batch_size{batch_size}.png'
                        .format(spar=spar, start=start, length=length, stacks=stacks, blocks=blocks, layers=layers, nHidden=nHidden, lr=self.learning_rate, train_len=train_len, batch_size=self.batch_size))

        plt.clf()

        shift = self.X_input_train[-1, -1, 0] - data_final['test_pred'][0, -1, 0]

        test_start_t = data_final['train_pred'].shape[0]
        x_lbl = 'x1'

        plt.plot(
            range(data_final['train_pred'].shape[0]),
            data_final['train_pred'][:, -1, 0] + shift,
            linestyle='--', linewidth=1, color='green', label='pred ' + x_lbl
        )

        plt.plot(
            range(data_final['train_pred'].shape[0]),
            data_final['train_true'][:, -1, 0],
            linestyle='-', linewidth=1, color='blue', label='true ' + x_lbl
        )

        plt.plot(
            range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
            data_final['test_pred'][:, -1, 0] + shift,
            linestyle='--', linewidth=1, color='red', label='pred ' + x_lbl
        )

        plt.plot(
            range(test_start_t, test_start_t + data_final['test_true'].shape[0]),
            data_final['test_true'][:, -1, 0],
            linestyle='-', linewidth=1, color='orange', label='true ' + x_lbl
        )


        plt.suptitle(
            "hl:{n_hidden}, epc:{epc}, stacks:{stacks}, blocks:{blocks}, layers:{layers}, lr:{learning_rate}, bs:{batch_size}, tl:{train_len}, err:{error}"
            .format(
                n_hidden=self.n_hidden,
                epc=self.epc,
                stacks=stacks,
                blocks=blocks,
                layers=layers,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                error=error,
                train_len=train_len
            )
        )

        if save:
            plt.savefig('NLPTrainTestFigures_spar{spar}_start{start}_length{length}_stacks{stacks}_blocks{blocks}_layers{layers}_hidden{nHidden}_lr{lr}_train_len{train_len}_batch_size{batch_size}.png'
                        .format(spar=spar, start=start, length=length, stacks=stacks, blocks=blocks, layers=layers, nHidden=nHidden, lr=self.learning_rate, train_len=train_len, batch_size=self.batch_size))



if __name__ == "__main__":
    model = NHITS(
        input_length=train_len,
        forecast_length=test_len,
        n_features=n_features
    )

    model.model.fit(
        X_input_train,
        X_output_train[:,:,0],
        epochs=100,
        batch_size=64
    )
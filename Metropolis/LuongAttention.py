import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pydot as pyd
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    concatenate, Activation, dot, LSTMCell, RNN, LayerNormalization, Embedding, MultiHeadAttention, Flatten, Reshape, Bidirectional, GlobalAveragePooling1D
from keras.models import Model
from keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import gather, sqrt

# from tensorflow.keras.layers import LayerNormalization


keras.utils.pydot = pyd


class NLPwithLSTM():
    def __init__(self, data, train_ratio=0.8, n_hidden=10, momentum=0.6, epc=100, learning_rate=0.01, batch_size=100, train_len=150, test_len=50, attention=""):
        self.train_ratio = train_ratio
        self.train_len = int(train_ratio * len(data))
        self.test_len = int((1-train_ratio) * len(data))
        self.x_index = np.linspace(0, 1, len(data), dtype=np.float32)
        self.data = data
        # self.data = self.simulatedData(len(data))
        self.attention = attention

        self.n_hidden = n_hidden
        self.momentum = momentum
        self.epc = epc
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.truncation_train_len = train_len
        self.truncation_test_len = test_len

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

        self.x_lbl = np.column_stack([x, self.x_index, [1]*self.train_len+[0]*(len(self.x_index)-self.train_len)])

        self.numberOfFeatures = 1 if (len(self.data.shape) == 1) else self.data.shape[1]

        self.x_train_max = np.abs(self.x_lbl[self.x_lbl[:, self.numberOfFeatures + 1] == 1, :1]).max(axis=0) * 1.2
        self.x_train_max = self.x_train_max.tolist()+[1]*2
        x_normalize = self.x_lbl #np.divide(self.x_lbl, self.x_train_max)



        # self.x_lbl has been replaced with x_normalize
        X_in, X_out, lbl = self.truncate(x_normalize,
                                         feature_cols=range(self.numberOfFeatures + 1),
                                         target_cols=range(self.numberOfFeatures + 1),
                                         label_col=self.numberOfFeatures + 1,
                                         train_len=self.truncation_train_len,
                                         test_len=self.truncation_test_len)

        self.X_input_train = X_in[np.where(lbl == 1)]
        self.X_output_train = X_out[np.where(lbl == 1)]
        self.X_input_test = X_in[np.where(lbl == 0)]
        self.X_output_test = X_out[np.where(lbl == 0)]

        # self.constructnetwork()
        # self.trainnetwork()
        # self.plotPredict()


    def simulatedData(self, l):
        n_ = l
        t = np.linspace(0, 50 * np.pi, n_)
        # pattern + trend + noise
        x1 = sum([20 * np.sin(i * t + np.pi) for i in range(5)]) + 0.01 * (t ** 2) + np.random.normal(0, 6, n_)
        x2 = sum([15 * np.sin(2 * i * t + np.pi) for i in range(5)]) + 0.5 * t + np.random.normal(0, 6, n_)
        # plt.figure(figsize=(15, 4))
        # plt.plot(range(len(x1)), x1, label='x1')
        # plt.plot(range(len(x2)), x2, label='x2')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        # plt.show()

        x_index = np.array(range(len(t)))
        x1_trend_param = np.polyfit(x_index[:self.train_len], x1[:self.train_len], 2)
        x2_trend_param = np.polyfit(x_index[:self.train_len], x2[:self.train_len], 1)
        print(x1_trend_param)
        print(x2_trend_param)
        x1_trend = (x_index**2)*x1_trend_param[0]+x_index*x1_trend_param[1]+x1_trend_param[2]
        x2_trend = x_index*x2_trend_param[0]+x2_trend_param[1]

        x1_detrend = x1 - x1_trend
        x2_detrend = x2 - x2_trend
        # plt.figure(figsize=(15, 4))    def truncate(self, x, feature_cols=range(3), target_cols=range(3), label_col=3, train_len=130, test_len=20):
        #         in_, out_, lbl = [], [], []
        #         for i in range(len(x)-train_len-test_len+1):
        #             in_.append(x[i:(i+train_len), feature_cols].tolist())
        #             out_.append(x[(i+train_len):(i+train_len+test_len), target_cols].tolist())
        #             lbl.append(x[i+train_len, label_col])
        #         return np.array(in_), np.array(out_), np.array(lbl)
        # plt.plot(range(len(x1_detrend)), x1_detrend, label='x2_detrend')
        # plt.plot(range(len(x2_detrend)), x2_detrend, label='x2_detrend')
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        # plt.show()

        # x_lbl = np.column_stack([x1_detrend, x2_detrend, x_index, [1] * self.train_len + [0] * (len(x_index) - self.train_len)])
        # print(x_lbl.shape)
        # print(x_lbl)

        # x_normalize = np.divide(x_lbl, self.x_train_max)
        # print(x_normalize)

        return x1_detrend




    def interlace(self, predicted, error):
        interlaced = np.zeros(predicted.shape[1] + predicted.shape[0],- 1)
        errorMatrix = np.zeros((interlaced.shape[0], predicted.shape[1]))
        summedInterlaced = np.zeros(interlaced.shape)
        errorArray = np.array(error)
        lenOfError = len(error)
        for i in range(errorMatrix.shape[1]):
            cappedLength = min(i + lenOfError, errorMatrix.shape[0])
            errorMatrix[i:cappedLength, i] = 1 / errorArray[:cappedLength - i]
        for i in range(predicted.shape[1]):
            interlaced[i:i + predicted.shape[0]] = predicted[:, i, 0]
            summedInterlaced += interlaced * errorMatrix[:, i]
        summedInterlaced /= np.sum(errorMatrix, axis=1)
        return summedInterlaced

    def create_model(layers, bidirectional=False):
        n_in_features = 1
        n_out_features = 1
        n_layers = len(layers)

        ## Encoder
        encoder_inputs = Input(shape=(None, n_in_features))
        lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]
        # if bidirectional:
        #     encoder = Bidirectional(RNN(lstm_cells, return_state=True))
        #     encoder_outputs_and_states = encoder(encoder_inputs)
        #     bi_encoder_states = encoder_outputs_and_states[1:]
        #     encoder_states = []
        #     for i in range(int(len(bi_encoder_states) / 2)):
        #         temp = concatenate([bi_encoder_states[i], bi_encoder_states[2 * n_layers + i]], axis=-1)
        #         encoder_states.append(temp)
        # else:
        encoder = RNN(lstm_cells, return_state=True)
        encoder_outputs_and_states = encoder(encoder_inputs)
        encoder_states = encoder_outputs_and_states[1:]

        ## Decoder
        decoder_inputs = Input(shape=(None, n_out_features))
        # if bidirectional:
        #     decoder_cells = [LSTMCell(hidden_dim * 2) for hidden_dim in layers]
        # else:
        decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]

        decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True)
        decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0]
        decoder_dense = Dense(n_out_features)
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    # We pass the hidden dimensions in each layer in the form of a list.
    # so length of the list becomes depth of the network.
    # model = create_model([60, 60], bidirectional=False)
    def constructnetwork(self):

        # ========= ENCODER =========
        encoder_inputs = Input(shape=(self.X_input_train.shape[1],
                                      self.X_input_train.shape[2]))

        encoder_lstm = LSTM(
            self.n_hidden,
            activation='tanh',
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        )

        enc_out, state_h, state_c = encoder_lstm(encoder_inputs)

        enc_out = LayerNormalization()(enc_out)
        state_h = LayerNormalization()(state_h)
        state_c = LayerNormalization()(state_c)

        # ========= DECODER (TRAINING) =========
        decoder_inputs = Input(shape=(self.X_output_train.shape[1],
                                      self.X_output_train.shape[2]))

        # IMPORTANT — shared dense layer
        self.decoder_dense = TimeDistributed(Dense(self.n_hidden))

        dec_processed = self.decoder_dense(decoder_inputs)

        decoder_lstm = LSTM(
            self.n_hidden,
            activation='tanh',
            dropout=0.1,
            recurrent_dropout=0.1,
            return_sequences=True,
            return_state=True,
            kernel_initializer="glorot_uniform",
            recurrent_initializer="orthogonal",
            bias_initializer="zeros",
        )

        dec_out, _, _ = decoder_lstm(
            dec_processed,
            initial_state=[state_h, state_c]
        )

        dec_out = LayerNormalization()(dec_out)

        # ========= ATTENTION =========
        attention = dot([dec_out, enc_out], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, enc_out], axes=[2, 1])
        context = LayerNormalization()(context)

        combined = concatenate([context, dec_out])

        decoder_outputs = TimeDistributed(
            Dense(self.X_output_train.shape[2])
        )(combined)

        # ========= TRAIN MODEL =========
        self.model = Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=0.5),
            loss="mse",
            metrics=["mae"]
        )

        # ========= ENCODER INFERENCE MODEL =========
        self.encoder_model = Model(
            encoder_inputs,
            [enc_out, state_h, state_c]
        )

        # ========= DECODER INFERENCE MODEL =========

        dec_state_input_h = Input(shape=(self.n_hidden,))
        dec_state_input_c = Input(shape=(self.n_hidden,))
        enc_out_input = Input(shape=(self.X_input_train.shape[1], self.n_hidden))
        dec_input_single = Input(shape=(1, self.X_output_train.shape[2]))

        # reuse SAME dense layer
        dec_single_processed = self.decoder_dense(dec_input_single)

        dec_out, h, c = decoder_lstm(
            dec_single_processed,
            initial_state=[dec_state_input_h, dec_state_input_c]
        )

        attn = dot([dec_out, enc_out_input], axes=[2, 2])
        attn = Activation("softmax")(attn)

        ctx = dot([attn, enc_out_input], axes=[2, 1])

        dec_combined = concatenate([ctx, dec_out])

        dec_pred = TimeDistributed(
            Dense(self.X_output_train.shape[2])
        )(dec_combined)

        self.decoder_model = Model(
            [dec_input_single, enc_out_input,
             dec_state_input_h, dec_state_input_c],
            [dec_pred, h, c]
        )

    def constructnetworkPatchTST(self):

        d_model = 64
        num_heads = 4
        num_layers = 4
        patch_len = 16

        seq_len = self.X_input_train.shape[1]
        n_features = self.X_input_train.shape[2]
        pred_len = self.X_output_train.shape[1]

        # =========================================================
        # INPUT
        # =========================================================
        inputs = Input(shape=(seq_len, n_features))

        # =========================================================
        # PATCHING
        # =========================================================
        num_patches = seq_len // patch_len

        x = Reshape(
            inputs,
            (-1, num_patches, patch_len * n_features)
        )

        # =========================================================
        # LINEAR PROJECTION
        # =========================================================
        x = Dense(d_model)(x)

        # =========================================================
        # POSITIONAL EMBEDDING
        # =========================================================
        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embed = Embedding(input_dim=num_patches, output_dim=d_model)(positions)
        x = x + pos_embed

        # =========================================================
        # TRANSFORMER ENCODER BLOCK
        # =========================================================
        def transformer_block(x):
            attn = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model
            )(x, x)

            x = LayerNormalization()(x + attn)

            ffn = Dense(d_model * 4, activation="gelu")(x)
            ffn = Dense(d_model)(ffn)

            x = LayerNormalization()(x + ffn)

            return x

        for _ in range(num_layers):
            x = transformer_block(x)

        # =========================================================
        # HEAD
        # =========================================================
        x = Flatten()(x)

        outputs = Dense(pred_len * n_features)(x)

        outputs = Reshape(outputs, (-1, pred_len, n_features))

        # =========================================================
        # MODEL
        # =========================================================
        self.model = Model(inputs, outputs)

        self.model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                clipnorm=1
            ),
            loss="mse",
            metrics=["mae"]
        )

    def constructnetworkBiderictionalEncoderTransformerDecoder(self):

        d_model = self.n_hidden
        n_heads = 4

        # =========================================================
        # ENCODER
        # =========================================================
        encoder_inputs = Input(shape=(self.X_input_train.shape[1],
                                      self.X_input_train.shape[2]))

        x = Bidirectional(
            LSTM(d_model, return_sequences=True)
        )(encoder_inputs)

        x = LayerNormalization()(x)

        encoder_outputs = Dense(d_model)(x)

        # =========================================================
        # DECODER INPUT
        # =========================================================
        decoder_inputs = Input(shape=(self.X_output_train.shape[1],
                                      self.X_output_train.shape[2]))

        y = Dense(d_model)(decoder_inputs)

        # =========================================================
        # TRANSFORMER DECODER BLOCK
        # =========================================================
        def transformer_block(y, enc):

            # Self attention
            attn1 = MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model
            )(y, y)

            y = LayerNormalization()(y + attn1)

            # Cross attention
            attn2 = MultiHeadAttention(
                num_heads=n_heads,
                key_dim=d_model
            )(y, enc)

            y = LayerNormalization()(y + attn2)

            # Feedforward
            f = Dense(d_model * 4, activation="relu")(y)
            f = Dense(d_model)(f)

            y = LayerNormalization()(y + f)

            return y

        # stack blocks
        for _ in range(2):
            y = transformer_block(y, encoder_outputs)

        # =========================================================
        # OUTPUT
        # =========================================================
        outputs = Dense(self.X_output_train.shape[2])(y)

        # =========================================================
        # TRAIN MODEL
        # =========================================================
        self.model = Model([encoder_inputs, decoder_inputs], outputs)

        self.model.compile(
            optimizer=Adam(
                learning_rate=self.learning_rate,
                clipnorm=1
            ),
            loss="mse",
            metrics=["mae"]
        )

        # =========================================================
        # ENCODER INFERENCE MODEL
        # =========================================================
        self.encoder_model = Model(encoder_inputs, encoder_outputs)

        # =========================================================
        # DECODER INFERENCE MODEL
        # =========================================================
        enc_out_input = Input(shape=encoder_outputs.shape[1:])

        dec_in_single = Input(shape=(1,
                                     self.X_output_train.shape[2]))

        y = Dense(d_model)(dec_in_single)

        for _ in range(2):
            y = transformer_block(y, enc_out_input)

        pred = Dense(self.X_output_train.shape[2])(y)

        self.decoder_model = Model(
            [dec_in_single, enc_out_input],
            pred
        )
    def constructnetwork_(self):
        self.input_train = Input(shape=(self.X_input_train.shape[1], self.X_input_train.shape[2]))#self.X_input_train.shape[2] - 1
        self.output_train = Input(shape=(self.X_output_train.shape[1], self.X_output_train.shape[2])) #self.X_output_train.shape[2] - 1

        # encoder
        if (self.attention == "Luong"):
            encoderLstm = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_state=True,
                return_sequences=True
            )
            encoder_stack_h, encoder_last_h, encoder_last_c = encoderLstm(self.input_train)

            encoder_stack_h = LayerNormalization()(encoder_stack_h)
            encoder_last_h = LayerNormalization()(encoder_last_h)
            encoder_last_c = LayerNormalization()(encoder_last_c)

            # decoder train
            self.decoder_dense = TimeDistributed(Dense(self.n_hidden))
            decoder_input = self.decoder_dense(self.output_train)
            decoderLstm = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_state=False,
                return_sequences=True
            )
            decoder_stack_h = decoderLstm(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
            decoder_stack_h = LayerNormalization()(decoder_stack_h)

            #attention train
            attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
            attention = Activation('softmax')(attention)

            context = dot([attention, encoder_stack_h], axes=[2,1])
            context = LayerNormalization()(context)
            # context = BatchNormalization(momentum=self.momentum)(context)

            decoder_combined_context = concatenate([context, decoder_stack_h])
            self.out = TimeDistributed(Dense(self.X_output_train.shape[2]))(decoder_combined_context)
        elif self.attention == "Recursive":
            encoder_lstm = LSTM(self.n_hidden, return_state=True, name='encoder_lstm')
            encoder_outputs, state_h, state_c = encoder_lstm(self.X_input_train)
            states = [state_h, state_c]
            model_encoder = Model(self.X_input_train, states)

            decoder_lstm = LSTM(self.n_hidden, return_sequences=True, return_state=True, name='decoder_lstm')
            decoder_dense = Dense(self.output_train.shape[2], activation='softmax')

            outputs, state_h, state_c = decoder_lstm(encoder_outputs, initial_state=states)

            encoder_lstm = LSTM(self.n_hidden, return_state=True, name='encoder_lstm')
            encoder_outputs, state_h, state_c = encoder_lstm(self.X_input_train)

            # initial context vector is the states of the encoder
            states = [state_h, state_c]

            # Set up the decoder layers
            # Attention: decoder receives 1 token at a time &
            # decoder outputs 1 token at a time

            decoder_lstm = LSTM(self.n_hidden, return_sequences=True,
                                return_state=True, name='decoder_lstm')
            decoder_dense = Dense(self.output_train.shape[2], activation='softmax', name='decoder_dense')

            all_outputs = []
            # Prepare decoder initial input data: just contains the START character 0
            # Note that we made it a constant one-hot-encoded in the model
            # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
            decoder_input_data = np.zeros((self.batch_size, 1, self.output_train.shape[2]))
            decoder_input_data[:, 0, 0] = 1

            # that is, [1 0 0 0 0 0 0 0 0 0] is the initial input for each loop
            inputs = decoder_input_data
            # decoder will only process one time step at a time
            # loops for fixed number of time steps: n_timesteps_in
            for _ in range(self.truncation_test_len):
                # Run the decoder on one time step
                outputs, state_h, state_c = decoder_lstm(inputs,
                                                         initial_state=states)
                outputs = decoder_dense(outputs)
                # Store the current prediction (we will concatenate all predictions later)
                all_outputs.append(outputs)
                # Reinject the outputs as inputs for the next loop iteration
                # as well as update the states
                inputs = outputs
                states = [state_h, state_c]

            # Concatenate all predictions such as [batch_size, timesteps, features]
            decoder_outputs = Lambda(lambda x: K.concatenate(x, axis=1))(all_outputs)

            # Define and compile model
            model = Model(self.input_train, decoder_outputs, name='model_encoder_decoder')
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

        elif self.attention == "None":
            # using multiple LSTM layers for encoding is not a problem at all.
            # Here I used 3. Pay attention to the flags. The sequence of the last
            # layer is not returned because we want a single vector that stores everything, not a time-sequence...
            encoder_input = Input(shape=(None, num_allowed_chars), name='encoder_input')
            encoder_lstm1 = LSTM(state_size, name='encoder_lstm1',
                                 return_sequences=True, return_state=True)
            encoder_lstm2 = LSTM(state_size, name='encoder_lstm2',
                                 return_sequences=True, return_state=True)
            encoder_lstm3 = LSTM(state_size, name='encoder_lstm3',
                                 return_sequences=False, return_state=True)

            # Connect all the LSTM-layers.
            x = encoder_input
            x, _, _ = encoder_lstm1(x)
            x, _, _ = encoder_lstm2(x)
            # only the states of the last layer are of interest.
            x, state_h, state_c = encoder_lstm3(x)
            encoder_output = x  # This is the encoded, fix-sized vector which seq2seq is all about
            encoder_states = [state_h, state_c]

            # here is something new: for every decoding layer we need an Input variable for both states hidden (h)
            # and cell state (c). Here I will use two stacked decoding layers and therefore initialize h1,c1,h2,c2.

            decoder_initial_state_h1 = Input(shape=(state_size,),
                                             name='decoder_initial_state_h1')

            decoder_initial_state_c1 = Input(shape=(state_size,),
                                             name='decoder_initial_state_c1')

            decoder_initial_state_h2 = Input(shape=(state_size,),
                                             name='decoder_initial_state_h2')

            decoder_initial_state_c2 = Input(shape=(state_size,),
                                             name='decoder_initial_state_c2')

            decoder_input = Input(shape=(None, num_allowed_chars), name='decoder_input')

            # pay attention of the return_sequence and return_state flags.
            decoder_lstm1 = LSTM(state_size, name='decoder_lstm1',
                                 return_sequences=True, return_state=True)
            decoder_lstm2 = LSTM(state_size, name='decoder_lstm2',
                                 return_sequences=True, return_state=True)

            decoder_dense = Dense(
                num_allowed_chars, activation='softmax', name="decoder_output")

            # connect the decoder for training (initial state = encoder_state)
            # I feed the encoder_states as inital input to both decoding lstm layers
            x = decoder_input
            x, h1, c1 = decoder_lstm1(x, initial_state=encoder_states)
            # I tried to pass [h1, c1] as initial states in line below, but that result in rubbish
            x, _, _ = decoder_lstm2(x, initial_state=encoder_states)
            decoder_output = decoder_dense(x)

            model_train = Model(inputs=[encoder_input, decoder_input],
                                outputs=decoder_output)

            model_encoder = Model(inputs=encoder_input,
                                  outputs=encoder_states)

            # this decoder model setup is used for inference
            # important! Every layer keeps its own states. This is, again, important in decode_sequence()
            x = decoder_input
            x, h1, c1 = decoder_lstm1(
                x, initial_state=[decoder_initial_state_h1, decoder_initial_state_c1])
            x, h2, c2 = decoder_lstm2(
                x, initial_state=[decoder_initial_state_h2, decoder_initial_state_c2])
            decoder_output = decoder_dense(x)
            decoder_states = [h1, c1, h2, c2]

            model_decoder = Model(
                inputs=[decoder_input] + [decoder_initial_state_h1, decoder_initial_state_c1,
                                          decoder_initial_state_h2, decoder_initial_state_c2],
                outputs=[decoder_output] + decoder_states)  # model outputs h1,c1,h2,c2!

            model_train.compile(optimizer='rmsprop',
                                loss='categorical_crossentropy', metrics=["acc"])

            plot_model(model_train, to_file=data_path_prefix +
                                            'spellchecker/model_train.png')
            plot_model(model_encoder, to_file=data_path_prefix +
                                              'spellchecker/model_encode.png')
            plot_model(model_decoder, to_file=data_path_prefix +
                                              'spellchecker/model_decode.png')

            def decode_sequence(input_seq, maxlen_decoder_sequence):
                # Encode the input as state vectors.
                initial_state = model_encoder.predict(input_seq)
                # I simply repeat the encoder states since
                # both decoding layers were trained on the encoded-vector
                # as initialization. I pass them into model_decoder.predict()
                initial_state = initial_state + initial_state

                # Generate empty target sequence of length 1.
                decoder_input_data = np.zeros((1, 1, num_allowed_chars))
                # Populate the first character of target sequence with the start character.
                decoder_input_data[0, 0, char_to_int['a']] = 1.

                # Sampling loop for a batch of sequences
                # (to simplify, here we assume a batch of size 1).
                stop_condition = False
                decoded_sentence = ''
                while not stop_condition:
                    # catch all returning states to feed them back in (see end of loop)
                    one_hot_char, h1, c1, h2, c2 = model_decoder.predict(
                        [decoder_input_data] + initial_state)

                    one_hot_char = one_hot_char[0][-1]

                    char_as_int = np.argmax(one_hot_char)
                    # print(char_as_int)
                    char_as_char = int_to_char[char_as_int]
                    decoded_sentence += char_as_char

                    # Exit condition: either hit max length or find stop character.
                    # (z is stop-char in this case)
                    if (char_as_char == 'z' or
                            len(decoded_sentence) >= maxlen_decoder_sequence):
                        stop_condition = True

                    # feed the predicted char back into next prediction step
                    decoder_input_data = np.zeros((1, 1, num_allowed_chars))
                    decoder_input_data[0, 0, char_as_int] = 1.

                    # Update states
                    initial_state = [h1, c1, h2, c2]

                return decoded_sentence

        else:
            # Encoder
            # encoder_inputs = Input(shape=(None, self.X_input_train.shape[2] - 1))
            # lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in [self.n_hidden, self.n_hidden]]
            # encoder = RNN(lstm_cells, return_state=True)
            # encoder_outputs_and_states = encoder(encoder_inputs)
            # encoder_outputs = encoder_outputs_and_states[0]
            # encoder_states = encoder_outputs_and_states[1:]
            #
            # ## Decoder
            # decoder_inputs = Input(shape=(None, self.X_output_train.shape[2] - 1))
            # decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in [self.n_hidden, self.n_hidden]]
            # # decoder_inputs = RepeatVector(self.output_train.shape[1])(encoder_outputs)
            # decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True)
            # decoder_outputs_and_states = decoder_lstm(decoder_inputs, initial_state=encoder_states)
            # decoder_outputs = decoder_outputs_and_states[0]
            # decoder_dense = Dense(self.X_output_train.shape[2])
            # decoder_outputs = decoder_dense(decoder_outputs)
            # self.out = decoder_outputs

            self.encoder_last_h1, self.encoder_last_h2, self.encoder_last_c = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=True,
                return_state=True
            )(self.input_train)

            self.encoder_last_h2 = BatchNormalization(momentum=self.momentum)(self.encoder_last_h2)
            self.encoder_last_c = BatchNormalization(momentum=self.momentum)(self.encoder_last_c)

            # self.encoder_last_h1_2, self.encoder_last_h2_2, self.encoder_last_c_2 = LSTM(
            #     self.n_hidden,
            #     activation='tanh',
            #     dropout=0.2,
            #     recurrent_dropout=0.2,
            #     return_sequences=True,
            #     return_state=True
            # )(self.encoder_last_h1)
            #
            # self.encoder_last_h2_2 = BatchNormalization(momentum=self.momentum)(self.encoder_last_h2_2)
            # self.encoder_last_c_2 = BatchNormalization(momentum=self.momentum)(self.encoder_last_c_2)

            self.encoder_last_h1_2, self.encoder_last_h2_2, self.encoder_last_c_2 = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_sequences=False,
                return_state=True
            )(self.encoder_last_h1)

            self.encoder_last_h2_2 = BatchNormalization(momentum=self.momentum)(self.encoder_last_h2_2)
            self.encoder_last_c_2 = BatchNormalization(momentum=self.momentum)(self.encoder_last_c_2)

            self.decoder_input = RepeatVector(self.output_train.shape[1])(self.encoder_last_h1_2)
            self.decoder = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_state=False,
                return_sequences=True
            )(self.decoder_input)#, initial_state=[self.encoder_last_h2, self.encoder_last_c])
            self.decoder = LSTM(
                self.n_hidden,
                activation='tanh',
                dropout=0.2,
                recurrent_dropout=0.2,
                return_state=False,
                return_sequences=True
            )(self.decoder, initial_state=[self.encoder_last_h2_2, self.encoder_last_c_2])

            # self.decoder = LSTM(
            #     self.n_hidden,
            #     activation='tanh',
            #     dropout=0.2,
            #     recurrent_dropout=0.2,
            #     return_state=False,
            #     return_sequences=True
            # )(self.decoder, initial_state=[self.encoder_last_h2_3, self.encoder_last_c_3])

            self.out = TimeDistributed(Dense(self.output_train.shape[2]))(self.decoder)

        self.model = Model(
            inputs=[self.input_train, self.output_train], #self.input_train,
            outputs=self.out
        )
        self.opt = Adam(
            learning_rate=self.learning_rate,
            clipnorm=1
        )
        self.model.compile(
            loss='mean_squared_error',
            optimizer=self.opt,
            metrics=['mae']
        )

        # Encoder model
        self.encoder_model = Model(
            self.input_train,
            [encoder_stack_h, encoder_last_h, encoder_last_c]
        )


        # Decoder inference model

        dec_state_input_h = Input(shape=(self.n_hidden,))
        dec_state_input_c = Input(shape=(self.n_hidden,))
        enc_out_input = Input(shape=(self.X_input_train.shape[1], self.n_hidden))
        dec_single_input = Input(shape=(1, self.numberOfFeatures))

        dec_out, dec_h, dec_c = decoderLstm(
            dec_single_input,
            initial_state=[dec_state_input_h, dec_state_input_c]
        )

        attn = dot([dec_out, enc_out_input], axes=[2,2])
        attn = Activation("softmax")(attn)
        ctx = dot([attn, enc_out_input], axes=[2,1])

        dec_comb = concatenate([ctx, dec_out])
        dec_pred = TimeDistributed(Dense(self.X_output_train.shape[2]))(dec_comb)

        self.decoder_model = Model(
            [
                dec_single_input,
                enc_out_input,
                dec_state_input_h,
                dec_state_input_c
            ],
            [dec_pred, dec_h, dec_c]
        )

    def trainnetwork(self):
        es = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=50
        )
        decoderInput = np.zeros_like(self.X_output_train[:, :, :self.numberOfFeatures + 1])
        decoderInput[:, 1:, :] = self.X_output_train[:, :-1, :]
        history = self.model.fit(
            [self.X_input_train[:, :, :self.numberOfFeatures + 1], decoderInput], #self.X_input_train[:, :, :self.numberOfFeatures + 1], #
            self.X_output_train[:, :, :self.numberOfFeatures + 1],
            validation_split=0.2,
            epochs=self.epc,
            verbose=0,
            callbacks=[es],
            batch_size=self.batch_size
        )
        # self.train_mae = abs(self.model.predict([self.X_input_train[:, :, :1], np.zeros(self.X_output_train[:, :, :1].shape)]) - self.X_output_train[:, :, :1]).mean(axis=(1, 2))
        # self.valid_mae = abs(self.model.predict([self.X_input_test[:, :, :1], np.zeros(self.X_output_test[:, :, :1].shape)]) - self.X_output_test[:, :, :1]).mean(axis=(1, 2))
        #
        # size = self.model.predict([self.X_input_test[:, :, :1], np.zeros(self.X_output_test[:, :, :1].shape)]).shape[0]
        # a = np.array([self.X_output_train[-1, :, 0] - self.model.predict([self.X_input_test[:, :, :1], np.zeros(self.X_output_test[:, :, :1].shape)])[0, :, 0], ] * size)[:, :, np.newaxis]
        # self.train_mae_shifted = abs(self.model.predict([self.X_input_train[:, :, :1], np.zeros(self.X_output_train[:, :, :1].shape)]) - self.X_output_train[:, :, :1]).mean(axis=(1, 2))
        # self.valid_mae_shifted = abs(self.model.predict([self.X_input_test[:, :, :1], np.zeros(self.X_output_test[:, :, :1].shape)]) + a - self.X_output_test[:, :, :1]).mean(axis=(1, 2))
        self.train_mae = (
            abs(self.predictTeacherForcing(self.X_input_train[:, :, :self.numberOfFeatures + 1])
                - self.X_output_train[:, :, :self.numberOfFeatures + 1]).mean(axis=(1, 2)))
        self.valid_mae = (
            abs(self.predictTeacherForcing(self.X_input_test[:, :, :self.numberOfFeatures + 1]) -
                self.X_output_test[:, :, :self.numberOfFeatures + 1]).mean(axis=(1, 2)))

        size = self.predictTeacherForcing(self.X_input_test[:, :, :self.numberOfFeatures + 1]).shape[0]
        a = (np.array([self.X_output_train[-1, :, 0] -
                       self.predictTeacherForcing(self.X_input_test[:, :, :self.numberOfFeatures + 1])[0, :, 0] ] * size)
             )[:, :, np.newaxis]
        self.train_mae_shifted = (
            abs(self.predictTeacherForcing(self.X_input_train[:, :, :self.numberOfFeatures + 1]) -
                self.X_output_train[:, :, :self.numberOfFeatures + 1]).mean(axis=(1, 2)))
        self.valid_mae_shifted = (
            abs(self.predictTeacherForcing(self.X_input_test[:, :, :self.numberOfFeatures + 1]) + a -
                self.X_output_test[:, :, :self.numberOfFeatures + 1]).mean(axis=(1, 2)))


        # self.train_mae = history.history['mae']
        # self.valid_mae = history.history['val_mae']

    def predictTeacherForcingBidirectionalEncoderTransformerDecoder(self, X):

        enc_out = self.encoder_model.predict(X, verbose=0)

        batch = X.shape[0]
        T_out = self.X_output_test.shape[1]
        n_feat = self.X_output_test.shape[2]

        decoder_input = np.zeros((batch, 1, n_feat))

        outputs = []

        for _ in range(T_out):
            pred = self.decoder_model.predict(
                [decoder_input, enc_out],
                verbose=0
            )

            outputs.append(pred)
            decoder_input = pred

        return np.concatenate(outputs, axis=1)
    def predictTeacherForcing(self, X):

        enc_out, h, c = self.encoder_model.predict(X, verbose=0)

        batch = X.shape[0]
        T_out = self.X_output_test.shape[1]
        n_feat = self.X_output_test.shape[2]

        decoder_input = np.zeros((batch, 1, n_feat))

        outputs = []

        for _ in range(T_out):
            pred, h, c = self.decoder_model.predict(
                [decoder_input, enc_out, h, c],
                verbose=0
            )

            outputs.append(pred)
            decoder_input = pred

        return np.concatenate(outputs, axis=1)
    def plotPredict(self, save=False, spar=1, start=1, length=1, inputLen=20, layers=3, nHidden=100, error=0, train_len = 230, figure=None):

        # train_pred_detrend = self.model.predict([self.X_input_train[:, :, :1], np.zeros(self.X_output_train[:, :, :1].shape)])*self.x_train_max[0]
        # test_pred_detrend = self.model.predict([self.X_input_test[:, :, :1], np.zeros(self.X_output_test[:, :, :1].shape)])*self.x_train_max[0]

        train_pred_detrend = self.predictTeacherForcing(self.X_input_train[:, :, :self.numberOfFeatures + 1])#*self.x_train_max[0]/10
        test_pred_detrend = self.predictTeacherForcing(self.X_input_test[:, :, :self.numberOfFeatures + 1])#*self.x_train_max[0]/10
        train_true_detrend = self.X_output_train[:, :, :self.numberOfFeatures + 1]#*self.x_train_max[0]
        test_true_detrend = self.X_output_test[:, :, :self.numberOfFeatures + 1]#*self.x_train_max[0]
        # train_pred_detrend = np.concatenate([train_pred_detrend, np.expand_dims(self.X_output_train[:, :, 1], axis=2)], axis=2)
        # test_pred_detrend = np.concatenate([test_pred_detrend, np.expand_dims(self.X_output_test[:, :, 1], axis=2)], axis=2)
        # train_true_detrend = np.concatenate([train_true_detrend, np.expand_dims(self.X_output_train[:, :, 1], axis=2)], axis=2)
        # test_true_detrend = np.concatenate([test_true_detrend, np.expand_dims(self.X_output_test[:, :, 1], axis=2)], axis=2)

        data_final = dict()
        for dt, lb in zip([train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
                          ['train_pred', 'train_true', 'test_pred', 'test_true']):
            dt_x1 = dt[:, :, 0]
            dt_x2 = dt[:, :, 1] #/ self.x_train_max[0]
            data_final[lb] = np.concatenate([np.expand_dims(dt_x1, axis=2), np.expand_dims(dt_x2, axis=2)], axis=2)

        ith_timestep = random.choice(range(data_final['train_pred'].shape[1]))
        plt.figure(figsize=(15, 5))
        train_start_t = 0
        test_start_t = data_final['train_pred'].shape[1]
        x_lbl = 'x1'
        # plt.plot(range(0, 0 + data_final['train_pred'].shape[0]),
        #          data_final['train_pred'][:, ith_timestep, 0],
        #          linestyle='--', linewidth=1, color='green', label='pred ' + x_lbl)
        # plt.plot(range(0, 0 + data_final['train_pred'].shape[0]),
        #          data_final['train_true'][:, ith_timestep, 0],
        #          linestyle='-', linewidth=1, color='blue', label='true ' + x_lbl)
        # plt.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
        #          data_final['test_pred'][:, ith_timestep, 0],
        #          linestyle='--', linewidth=1, color='red', label='pred ' + x_lbl)
        # plt.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
        #          data_final['test_true'][:, ith_timestep, 0],
        #          linestyle='-', linewidth=1, color='orange', label='true ' + x_lbl)

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
        # plt.plot(data_final['test_true'][-1, :, 0],
        #          data_final['test_pred'][-1, :, 0],
        #          'x')
        # for lb, tm, clrs in zip(['train', 'test'], [train_start_t, test_start_t],
        #                         [['green', 'red'], ['blue', 'orange']]):
        #     for i, x_lbl in zip([0], ['x1']):
        #         plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
        #                  data_final[lb + '_pred'][:, ith_timestep, i],
        #                  linestyle='--', linewidth=1, color=clrs[0], label='pred ' + x_lbl)
        #         plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
        #                  data_final[lb + '_true'][:, ith_timestep, i],
        #                  linestyle='-', linewidth=1, color=clrs[1], label='true ' + x_lbl)

        plt.title('{}th time step in all samples'.format(ith_timestep))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=8)
        if save:
            plt.savefig('NLPtestFigures_spar{spar}_start{start}_length{length}_layers{layers}_hidden{nHidden}_lr{lr}_train_len{train_len}_batch_size{batch_size}.png'
                        .format(spar=spar, start=start, length=length, layers=layers, nHidden=nHidden, lr=self.learning_rate, train_len=train_len, batch_size=self.batch_size))

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
        # plt.plot(data_final['test_true'][:, -1, 0],
        #          data_final['test_pred'][:, -1, 0],
        #          'x')
        # for i in range(20):
        #     test_start_t = data_final['train_pred'].shape[0]
        #     x_lbl = 'x1'
        #     plt.subplot(4, 5, i+1)
        #     plt.plot(range(0, 0 + data_final['train_pred'].shape[0]),
        #              data_final['train_pred'][:, i, 0],
        #              linestyle='--', linewidth=1, color='green', label='pred ' + x_lbl)
        #     plt.plot(range(0, 0 + data_final['train_pred'].shape[0]),
        #              data_final['train_true'][:, i, 0],
        #              linestyle='-', linewidth=1, color='blue', label='true ' + x_lbl)
        #     plt.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
        #              data_final['test_pred'][:, i, 0],
        #              linestyle='--', linewidth=1, color='red', label='pred ' + x_lbl)
        #     plt.plot(range(test_start_t, test_start_t + data_final['test_pred'].shape[0]),
        #              data_final['test_true'][:, i, 0],
        #              linestyle='-', linewidth=1, color='orange', label='true ' + x_lbl)


        plt.suptitle(
            "hl:{n_hidden}, mom:{momentum}, epc:{epc}, lr:{learning_rate}, bs:{batch_size}, tl:{train_len}, err:{error}"
            .format(
                n_hidden=self.n_hidden,
                momentum=self.momentum,
                epc=self.epc,
                learning_rate=self.learning_rate,
                batch_size=self.batch_size,
                error=error,
                train_len=train_len
            )
        )

        if save:
            plt.savefig('NLPTrainTestFigures_spar{spar}_start{start}_length{length}_layers{layers}_hidden{nHidden}_lr{lr}_train_len{train_len}_batch_size{batch_size}.png'
                        .format(spar=spar, start=start, length=length, layers=layers, nHidden=nHidden, lr=self.learning_rate, train_len=train_len, batch_size=self.batch_size))

    def predict(self):

        train_pred_detrend = self.model.predict(self.X_input_train[:, :, :2])#*x_train_max[:1]
        test_pred_detrend = self.model.predict(self.X_input_test[:, :, :2])#*x_train_max[:1]
        train_true_detrend = self.X_output_train[:, :, :2]#*x_train_max[:1]
        test_true_detrend = self.X_output_test[:, :, :1]#*x_train_max[:1]
        train_pred_detrend = np.concatenate([train_pred_detrend, np.expand_dims(self.X_output_train[:, :, 1], axis=2)], axis=2)
        test_pred_detrend = np.concatenate([test_pred_detrend, np.expand_dims(self.X_output_test[:, :, 1], axis=2)], axis=2)
        train_true_detrend = np.concatenate([train_true_detrend, np.expand_dims(self.X_output_train[:, :, 1], axis=2)], axis=2)
        test_true_detrend = np.concatenate([test_true_detrend, np.expand_dims(self.X_output_test[:, :, 1], axis=2)], axis=2)


        data_final = dict()
        for dt, lb in zip([train_pred_detrend, train_true_detrend, test_pred_detrend, test_true_detrend],
                          ['train_pred', 'train_true', 'test_pred', 'test_true']):
            dt_x1 = dt[:, :, 0]
            # dt_x2 = dt[:, :, 1]
            data_final[lb] = np.expand_dims(dt_x1, axis=2)#np.concatenate(
            # [np.expand_dims(dt_x1, axis=2), np.expand_dims(dt_x2, axis=2)], axis=2)

        for lb in ['train', 'test']:
            MAE_overall = abs(data_final[lb+'_pred'] - data_final[lb+'_true']).mean()
            MAE_ = abs(data_final[lb+'_pred'] - data_final[lb+'_true']).mean(axis=(1, 2))
            plt.figure(figsize=(15, 3))
            plt.plot(MAE_)
            plt.title('MAE '+lb+': overall MAE = '+str(MAE_overall))
            plt.show()

        ith_timestep = random.choice(range(data_final[lb + '_pred'].shape[1]))
        plt.figure(figsize=(15, 5))
        train_start_t = 0
        test_start_t = data_final['train_pred'].shape[0]
        for lb, tm, clrs in zip(['train', 'test'], [train_start_t, test_start_t], [['green', 'red'], ['blue', 'orange']]):
            for i, x_lbl in zip([0], ['x1']):
                plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
                         data_final[lb + '_pred'][:, ith_timestep, i],
                         linestyle='--', linewidth=1, color=clrs[0], label='pred ' + x_lbl)
                plt.plot(range(tm, tm + data_final[lb + '_pred'].shape[0]),
                         data_final[lb + '_true'][:, ith_timestep, i],
                         linestyle='-', linewidth=1, color=clrs[1], label='true ' + x_lbl)

        plt.title('{}th time step in all samples'.format(ith_timestep))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=8)
        plt.show()

        lb = 'test'
        plt.figure(figsize=(15, 5))
        for i, x_lbl, clr in zip([0], ['x1'], ['green']):
            plt.plot(data_final[lb+'_pred'][:, ith_timestep, i], linestyle='--', color=clr, label='pred '+x_lbl)
            plt.plot(data_final[lb+'_true'][:, ith_timestep, i], linestyle='-', color=clr, label='true '+x_lbl)
        plt.title('({}): {}th time step in all samples'.format(lb, ith_timestep))
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=2)
        plt.show()

        return data_final

    def constructLuong(self):
        #Luong Attention
        input_train = Input(shape=(self.X_input_train.shape[1], self.X_input_train.shape[2]-1))
        output_train = Input(shape=(self.X_output_train.shape[1], self.X_output_train.shape[2]-1))
        encoder_stack_h, encoder_last_h, encoder_last_c = LSTM(
            self.n_hidden,
            activation='elu',
            dropout=0.2,
            recurrent_dropout=0.2,
            return_state=True,
            return_sequences=True
        )(input_train)

        encoder_last_h = BatchNormalization(momentum=self.momentum)(encoder_last_h)
        encoder_last_c = BatchNormalization(momentum=self.momentum)(encoder_last_c)

        decoder_input = RepeatVector(output_train.shape[1])(encoder_last_h)
        decoder_stack_h = LSTM(
            self.hidden,
            activation='elu',
            dropout=0.2,
            recurrent_dropout=0.2,
            return_state=False,
            return_sequences=True
        )(decoder_input, initial_state=[encoder_last_h, encoder_last_c])

        attention = dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
        attention = Activation('softmax')(attention)

        context = dot([attention, encoder_stack_h], axes=[2,1])
        context = BatchNormalization(momentum=self.momentum)(context)

        decoder_combined_context = concatenate([context, decoder_stack_h])
        out = TimeDistributed(Dense(output_train.shape[2]))(decoder_combined_context)

        model = Model(inputs=input_train, outputs=out)
        opt = Adam(learning_rate=self.learning_rate, clipnorm=1)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])


    def createPredictInput(self, data, train_ratio, truncation_train_len, truncation_test_len):
        x_index = np.array(range(len(data)))
        train_len = int(train_ratio * len(data))
        x_lbl = np.column_stack([data, x_index, [1]*train_len+[0]*(len(x_index)-train_len)])
        X_in, X_out, lbl = self.truncate(
            x=x_lbl,
            feature_cols=range(2),
            target_cols=range(2),
            label_col=2,
            train_len=truncation_train_len,
            test_len=truncation_test_len
        )
        return X_in[np.where(lbl == 0)]

    def build_diffusion_schedule(self, T = 1000):

        beta_start = 1e-4
        beta_end = 0.02

        self.T = T

        self.betas = np.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.alpha_hat = np.cumprod(self.alphas)


    def q_sample(self, y0, t, noise):

        alpha_hat_t = gather(self.alpha_hat, t)

        return (
                sqrt(alpha_hat_t) * y0 +
                sqrt(1 - alpha_hat_t) * noise
        )

    def constructnetworkDARTModel(self):

        d_model = 64
        num_heads = 4
        num_layers = 4

        seq_len = self.X_input_train.shape[1]
        pred_len = self.X_output_train.shape[1]
        n_features = self.X_input_train.shape[2]

        # =========================================================
        # INPUTS
        # =========================================================
        history_input = Input(shape=(seq_len, n_features))
        future_noisy = Input(shape=(pred_len, n_features))
        t_input = Input(shape=(1,), dtype=tf.int32)

        # =========================================================
        # ENCODER (History)
        # =========================================================
        x = Dense(d_model)(history_input)

        for _ in range(3):
            attn = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model
            )(x, x)
            x = LayerNormalization()(x + attn)

            f = Dense(d_model * 4, activation="gelu")(x)
            f = Dense(d_model)(f)
            x = LayerNormalization()(x + f)

        history_context = GlobalAveragePooling1D()(x)
        history_context = Dense(d_model)(history_context)

        # =========================================================
        # TIME EMBEDDING
        # =========================================================
        t_embed = Embedding(1000, d_model)(t_input)
        t_embed = Flatten()(t_embed)

        conditioning = history_context + t_embed
        conditioning = tf.expand_dims(conditioning, 1)

        # =========================================================
        # DIFFUSION TRANSFORMER (Predict Noise)
        # =========================================================
        y = Dense(d_model)(future_noisy)

        y = y + conditioning

        for _ in range(num_layers):
            attn = MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model
            )(y, y)
            y = LayerNormalization()(y + attn)

            f = Dense(d_model * 4, activation="gelu")(y)
            f = Dense(d_model)(f)
            y = LayerNormalization()(y + f)

        noise_pred = Dense(n_features)(y)

        self.model = Model(
            [history_input, future_noisy, t_input],
            noise_pred
        )

        self.model.compile(
            optimizer=Adam(1e-4),
            loss="mse"
        )


    def train_diffusion(self):

        T = self.T
        batch_size = self.batch_size

        for epoch in range(self.epc):

            idx = np.random.randint(0, self.X_input_train.shape[0], batch_size)

            history = self.X_input_train[idx]
            future = self.X_output_train[idx]

            t = np.random.randint(0, T, size=(batch_size, 1))

            noise = np.random.normal(size=future.shape)

            noisy_future = self.q_sample(future, t[:, 0], noise)

            self.model.train_on_batch(
                [history, noisy_future, t],
                noise
            )

            if epoch % 10 == 0:
                print("Epoch:", epoch)

    def sample_forecast(self, history):

        batch = history.shape[0]
        pred_len = self.X_output_train.shape[1]
        n_features = self.X_output_train.shape[2]

        x = np.random.normal(size=(batch, pred_len, n_features))

        for t in reversed(range(self.T)):

            t_tensor = np.full((batch, 1), t)

            noise_pred = self.model.predict(
                [history, x, t_tensor],
                verbose=0
            )

            alpha = self.alphas[t]
            alpha_hat = self.alpha_hat[t]
            beta = self.betas[t]

            x = (
                    1 / np.sqrt(alpha) *
                    (x - (1 - alpha) / np.sqrt(1 - alpha_hat) * noise_pred)
            )

            if t > 0:
                x += np.sqrt(beta) * np.random.normal(size=x.shape)

        return x
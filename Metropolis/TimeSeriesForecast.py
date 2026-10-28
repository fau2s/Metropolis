import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Input, BatchNormalization, \
    concatenate, Activation, dot, LSTMCell, RNN, LayerNormalization, Embedding, MultiHeadAttention, Flatten, Reshape, Bidirectional, GlobalAveragePooling1D
from keras.models import Model

class TimeSeriesForecaster:

    def __init__(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="luong",
        d_model=64,
        n_hidden=64,
        num_heads=4,
        num_layers=3,
        learning_rate=1e-3
    ):

        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        self.model_type = model_type
        self.d_model = d_model
        self.n_hidden = n_hidden
        self.num_heads = num_heads
        self.num_layers = num_layers

        self.learning_rate = learning_rate

        self.seq_len = X_train.shape[1]
        self.pred_len = y_train.shape[1]
        self.n_features = X_train.shape[2]

        self.build_model()

    # =========================================================
    # MAIN SWITCH
    # =========================================================

    def build_model(self):

        if self.model_type == "lstm":
            self.model = self.build_seq2seq()

        elif self.model_type == "luong":
            self.model = self.build_luong_attention()

        elif self.model_type == "patchtst":
            self.model = self.build_patchtst()

        elif self.model_type == "transformer":
            self.model = self.build_transformer()

        elif self.model_type == "dart":
            self.model = self.build_dart()

        else:
            raise ValueError("Unknown model type")

        self.model.compile(
            optimizer=Adam(self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )

        self.model.summary()

    # =========================================================
    # 1️⃣ SEQ2SEQ
    # =========================================================

    def build_seq2seq(self):

        enc_in = Input(shape=(self.seq_len, self.n_features))
        enc_out, h, c = LSTM(self.n_hidden, return_state=True)(enc_in)

        dec_in = RepeatVector(self.pred_len)(enc_out)
        dec_out = LSTM(self.n_hidden, return_sequences=True)(dec_in)

        outputs = Dense(self.n_features)(dec_out)

        return Model(enc_in, outputs)

    # =========================================================
    # 2️⃣ LUONG ATTENTION
    # =========================================================

    def build_luong_attention(self):

        enc_in = Input(shape=(self.seq_len, self.n_features))
        enc_seq, h, c = LSTM(
            self.n_hidden,
            return_sequences=True,
            return_state=True
        )(enc_in)

        dec_in = RepeatVector(self.pred_len)(h)
        dec_seq = LSTM(
            self.n_hidden,
            return_sequences=True
        )(dec_in, initial_state=[h, c])

        attention = dot(axes=[2, 2])([dec_seq, enc_seq])
        attention = Activation("softmax")(attention)

        context = dot(axes=[2, 1])([attention, enc_seq])

        combined = concatenate()([context, dec_seq])

        outputs = Dense(self.n_features)(combined)

        return Model(enc_in, outputs)

    # =========================================================
    # 3️⃣ PATCHTST
    # =========================================================

    def build_patchtst(self):

        patch_len = 16
        num_patches = self.seq_len // patch_len

        inputs = Input(shape=(self.seq_len, self.n_features))

        x = Reshape((num_patches, patch_len * self.n_features))(inputs)
        x = Dense(self.d_model)(x)

        for _ in range(self.num_layers):
            attn = MultiHeadAttention(
                self.num_heads,
                key_dim=self.d_model
            )(x, x)
            x = LayerNormalization()(x + attn)

            f = Dense(self.d_model * 4, activation="gelu")(x)
            f = Dense(self.d_model)(f)
            x = LayerNormalization()(x + f)

        x = Flatten()(x)
        x = Dense(self.pred_len * self.n_features)(x)
        outputs = Reshape((self.pred_len, self.n_features))(x)

        return Model(inputs, outputs)

    # =========================================================
    # 4️⃣ TRANSFORMER ENCODER
    # =========================================================

    def build_transformer(self):

        inputs = Input(shape=(self.seq_len, self.n_features))
        x = Dense(self.d_model)(inputs)

        for _ in range(self.num_layers):
            attn = MultiHeadAttention(
                self.num_heads,
                key_dim=self.d_model
            )(x, x)
            x = LayerNormalization()(x + attn)

            f = Dense(self.d_model * 4, activation="gelu")(x)
            f = Dense(self.d_model)(f)
            x = LayerNormalization()(x + f)

        x = GlobalAveragePooling1D()(x)
        x = Dense(self.pred_len * self.n_features)(x)
        outputs = Reshape((self.pred_len, self.n_features))(x)

        return Model(inputs, outputs)

    # =========================================================
    # 5️⃣ DART (DIFFUSION)
    # =========================================================

    def build_dart(self):

        history = Input(shape=(self.seq_len, self.n_features))
        noisy_future = Input(shape=(self.pred_len, self.n_features))
        t_input = Input(shape=(1,), dtype=tf.int32)

        x = Dense(self.d_model)(history)

        for _ in range(3):
            attn = MultiHeadAttention(
                self.num_heads,
                key_dim=self.d_model
            )(x, x)
            x = LayerNormalization()(x + attn)

            f = Dense(self.d_model * 4, activation="gelu")(x)
            f = Dense(self.d_model)(f)
            x = LayerNormalization()(x + f)

        context = GlobalAveragePooling1D()(x)

        t_embed = Embedding(1000, self.d_model)(t_input)
        t_embed = Flatten()(t_embed)

        cond = context + t_embed
        cond = tf.expand_dims(cond, 1)

        y = Dense(self.d_model)(noisy_future)
        y = y + cond

        for _ in range(self.num_layers):
            attn = MultiHeadAttention(
                self.num_heads,
                key_dim=self.d_model
            )(y, y)
            y = LayerNormalization()(y + attn)

            f = Dense(self.d_model * 4, activation="gelu")(y)
            f = Dense(self.d_model)(f)
            y = LayerNormalization()(y + f)

        noise_pred = Dense(self.n_features)(y)

        return Model([history, noisy_future, t_input], noise_pred)

    # =========================================================
    # TRAIN
    # =========================================================

    def train(self, epochs=50, batch_size=32):

        if self.model_type != "dart":

            self.model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_test, self.y_test),
                epochs=epochs,
                batch_size=batch_size
            )

        else:
            print("Use diffusion training loop separately.")
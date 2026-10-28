import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, Model, Input



def sliding_rfft_matrix(ts, L, step=1, sampling_rate=1.0):
    """
    Compute rFFT for sliding windows.
    Returns:
      S_complex: np.ndarray shape (n_freqs, n_windows), complex-valued
      times_center: array of center positions (in samples)
      freqs: frequency bins (Hz)
    """
    ts = np.asarray(ts)
    n = len(ts)
    starts = list(range(0, n - L + 1, step))
    n_windows = len(starts)
    # rfft produces n_freqs = L//2 + 1 complex bins
    n_freqs = L // 2 + 1
    S = np.zeros((n_freqs, n_windows), dtype=np.complex64)
    times_center = []
    for i, s in enumerate(starts):
        w = ts[s:s+L]
        fft_vals = np.fft.rfft(w)  # complex values length n_freqs
        S[:, i] = fft_vals
        times_center.append(s + L // 2)
    freqs = np.fft.rfftfreq(L, d=1.0 / sampling_rate)
    return S, np.array(times_center), freqs

def complex_matrix_to_real_features(S_complex):
    """
    Convert complex matrix (n_freqs x n_windows) to real-feature matrix
    with shape (n_windows, 2*n_freqs) with ordering [Re..., Im...].
    """
    real = np.real(S_complex).T  # shape (n_windows, n_freqs)
    imag = np.imag(S_complex).T  # shape (n_windows, n_freqs)
    return np.concatenate([real, imag], axis=1)  # (n_windows, 2*n_freqs)

def real_features_to_complex_vector(vec):
    """
    Convert a single feature vector of length 2*n_freqs into complex vector length n_freqs.
    """
    n2 = vec.shape[-1]
    assert n2 % 2 == 0
    n_freqs = n2 // 2
    real = vec[:n_freqs]
    imag = vec[n_freqs:]
    return real + 1j * imag

def make_seq2seq_samples(feature_matrix, in_len, out_len):
    """
    feature_matrix: shape (n_windows, n_features)
    produce X (n_samples, in_len, n_features) and
            Y (n_samples, out_len, n_features)
    """
    n_windows = feature_matrix.shape[0]
    samples = []
    targets = []
    for s in range(0, n_windows - in_len - out_len + 1):
        X = feature_matrix[s : s + in_len]
        Y = feature_matrix[s + in_len : s + in_len + out_len]
        samples.append(X)
        targets.append(Y)
    return np.array(samples), np.array(targets)

def build_seq2seq_model(n_features, in_len, out_len, latent_dim=128):
    """
    Build a basic encoder-decoder LSTM Seq2Seq model with teacher forcing during training.
    Encoder: LSTM -> returns states
    Decoder: LSTM (return_sequences=True) -> Dense to n_features
    """
    # Encoder
    encoder_inputs = Input(shape=(in_len, n_features), name='encoder_inputs')
    encoder_lstm = layers.LSTM(latent_dim, return_state=True, name='encoder_lstm')
    _, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Decoder
    decoder_inputs = Input(shape=(out_len, n_features), name='decoder_inputs')  # teacher forcing inputs
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = layers.TimeDistributed(layers.Dense(n_features), name='decoder_dense')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs, name='seq2seq_model')
    model.compile(optimizer='adam', loss='mse')
    return model

# def make_decoder_inference_models(model, latent_dim, n_features):
#     """
#     Create separate encoder and decoder models for inference (prediction).
#     """
#     # Extract layers by name
#     encoder_inputs = model.get_layer('encoder_inputs').input
#     encoder_lstm = model.get_layer('encoder_lstm')
#     _, state_h_enc, state_c_enc = encoder_lstm(encoder_inputs)
#     encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])
#
#     # Decoder setup
#     decoder_inputs = Input(shape=(1, n_features), name='decoder_input_infer')  # single time-step input
#     dec_state_h = Input(shape=(latent_dim,), name='dec_state_h')
#     dec_state_c = Input(shape=(latent_dim,), name='dec_state_c')
#     decoder_lstm = model.get_layer('decoder_lstm')
#     decoder_dense = model.get_layer('decoder_dense')
#
#     dec_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=[dec_state_h, dec_state_c])
#     dec_outputs = decoder_dense(dec_outputs)
#     decoder_model = Model([decoder_inputs, dec_state_h, dec_state_c], [dec_outputs, state_h, state_c])
#     return encoder_model, decoder_model
def make_decoder_inference_models(model, latent_dim, n_features):
    # Extract trained layers from the full model
    encoder_inputs = model.input[0]  # encoder input layer
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # encoder LSTM outputs
    encoder_states = [state_h_enc, state_c_enc]

    encoder_model = Model(encoder_inputs, encoder_states)

    # Reconstruct the decoder for inference
    decoder_inputs = Input(shape=(1, n_features))
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]

    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)

    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states
    )

    return encoder_model, decoder_model

def seq2seq_predict(encoder_model, decoder_model, input_seq, out_len, n_features):
    """
    Given trained inference models, predict out_len steps.
    input_seq: shape (1, in_len, n_features)
    Returns predicted array shape (out_len, n_features)
    """
    # Encode
    states = encoder_model.predict(input_seq, verbose=0)
    # First decoder input: zeros (or last real value). We'll use zeros.
    next_input = np.zeros((1, 1, n_features), dtype=np.float32)
    pred_seq = []
    h, c = states[0], states[1]
    for _ in range(out_len):
        dec_out, h, c = decoder_model.predict([next_input, h, c], verbose=0)
        # dec_out shape (1,1,n_features)
        pred = dec_out[0, 0, :]
        pred_seq.append(pred.copy())
        # prepare next input: here feed predicted (no teacher forcing)
        next_input = dec_out  # shape (1,1,n_features)
    return np.array(pred_seq)  # (out_len, n_features)


# -------------------------
# Example end-to-end flow
# -------------------------
if __name__ == "__main__":
    # --- synthetic test signal ---
    sr = 100  # sampling rate
    T = 5000
    t = np.arange(T) / sr
    # two-tone + chirp + noise (complex enough)
    signal = 0.8 * np.sin(2*np.pi*2*t) + 0.5 * np.sin(2*np.pi*10*t * (1 + 0.01*t)) + 0.2 * np.random.randn(len(t))

    # parameters
    L = 256               # FFT window length (samples)
    step = 1             # sliding step (samples)
    in_len = 100           # number of windows used by encoder
    out_len = 30           # K: forecast horizon in windows
    latent_dim = 128
    batch_size = 32
    epochs = 40

    # --- compute sliding rFFT matrix ---
    S_complex, centers, freqs = sliding_rfft_matrix(signal, L=L, step=step, sampling_rate=sr)
    # S_complex shape: (n_freqs, n_windows)
    n_freqs, n_windows = S_complex.shape
    print("n_freqs, n_windows:", n_freqs, n_windows)

    # convert to real features
    feature_mat = complex_matrix_to_real_features(S_complex)  # shape (n_windows, 2*n_freqs)
    n_features = feature_mat.shape[1]

    # scale features (important)
    scaler = StandardScaler()
    feature_mat_scaled = scaler.fit_transform(feature_mat)

    # make seq2seq samples
    X, Y = make_seq2seq_samples(feature_mat_scaled, in_len, out_len)
    print("X shape, Y shape:", X.shape, Y.shape)  # (n_samples, in_len, n_features) (n_samples, out_len, n_features)


    # Prepare decoder input for training: teacher forcing -> decoder_inputs are Y shifted right with zeros at start
    decoder_inputs = np.zeros_like(Y)
    decoder_inputs[:, 1:, :] = Y[:, :-1, :]  # shift right
    # decoder_inputs[:, 0, :] remains zeros

    X_train, X_val, decoder_inputs_train, decoder_inputs_val, Y_train, Y_val = train_test_split(
        X, decoder_inputs, Y, test_size=0.1, shuffle=False
    )


    # Build and train model
    model = build_seq2seq_model(n_features=n_features, in_len=in_len, out_len=out_len, latent_dim=latent_dim)
    model.summary()
    # history = model.fit([X, decoder_inputs], Y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    history = model.fit(
        [X_train, decoder_inputs_train],
        Y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=([X_val, decoder_inputs_val], Y_val)
    )
    # --- Forecast using the last in_len windows in the data ---
    last_input = feature_mat_scaled[-in_len:]  # shape (in_len, n_features)
    last_input = last_input[np.newaxis, :, :]  # (1, in_len, n_features)

    # create encoder/decoder inference models
    encoder_model, decoder_model = make_decoder_inference_models(model, latent_dim=latent_dim, n_features=n_features)

    # predict out_len steps (sequence of predicted feature vectors)
    # --- Generate predictions with Seq2Seq model ---
    preds = seq2seq_predict(encoder_model, decoder_model, last_input, out_len=out_len, n_features=n_features)
    print("preds shape:", preds.shape)  # (out_len, n_features)

    # --- Take the Kth (last) forecasted FFT components ---
    kth_pred_features = preds[-1]  # (n_features,)
    kth_pred_unscaled = scaler.inverse_transform(kth_pred_features.reshape(1, -1))[0]
    kth_pred_complex = real_features_to_complex_vector(kth_pred_unscaled)

    # --- Get the real FFT of the last observed window ---
    last_obs_window = signal[-L:]
    last_obs_fft = np.fft.rfft(last_obs_window)

    # --- Convert both FFTs to magnitude & phase for comparison ---
    freqs = np.fft.rfftfreq(L, d=1 / sr)
    mag_real_real = np.real(last_obs_fft)
    mag_real_imag = np.imag(last_obs_fft)
    mag_pred_real = np.real(kth_pred_complex)
    mag_pred_imag = np.imag(kth_pred_complex)
    abs_pred = np.abs(kth_pred_complex)
    abs_real = np.abs(last_obs_fft)
    phase_real = np.angle(last_obs_fft)
    phase_pred = np.angle(kth_pred_complex)

    # --- Inverse FFT of predicted frequencies to reconstruct time domain ---
    time_pred_window = np.fft.irfft(kth_pred_complex, n=L)

    # -----------------------------
    # Forecast next FFT window and reconstruct
    # -----------------------------
    last_input = X[-1:].copy()
    preds = seq2seq_predict(encoder_model, decoder_model, last_input, out_len=out_len, n_features=n_features)
    print("Preds shape:", preds.shape)

    kth_pred_features = preds[-1]
    kth_pred_unscaled = scaler.inverse_transform(kth_pred_features.reshape(1, -1))[0]
    kth_pred_complex = real_features_to_complex_vector(kth_pred_unscaled)
    time_pred_window = np.fft.irfft(kth_pred_complex, n=L)

    # -----------------------------
    # Plot time-domain predictions
    # -----------------------------
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(L) / sr, signal[-L:], label='Last observed window')
    plt.plot(np.arange(L) / sr, time_pred_window.real, label='Predicted window')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Comparison of last observed and predicted time-domain window')
    plt.show()

    # -----------------------------
    # Plot frequency-domain predictions (low frequencies)
    # -----------------------------
    last_fft = np.fft.rfft(signal[-L:])
    freqs = np.fft.rfftfreq(L, 1 / sr)

    plt.figure(figsize=(10, 5))
    plt.plot(freqs[:40], np.abs(last_fft[:40]), label='True FFT (last window)')
    plt.plot(freqs[:40], np.abs(kth_pred_complex[:40]), label='Predicted FFT (Kth step)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.title('Comparison of true vs predicted low-frequency magnitudes')
    plt.show()

    # -----------------------------
    # Plot training history
    # -----------------------------
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()

# -----------------------------
# Also show single-window comparison (last observed vs first predicted reconstructed window)
# -----------------------------
# Reconstruct the first predicted window individually for comparison
    preds_unscaled = scaler.inverse_transform(preds)  # (out_len, n_features)

    # prepare Hann synthesis window for overlap-add
    synth_win = np.hanning(L)
    # arrays for accumulating predicted signal and normalization (overlap count)
    # choose forecast length: start last observed window at last_obs_start; we will forecast out_len windows, each shifted by 'step'
    last_window_start = len(signal) - L  # index of last observed window start in original signal
    # forecasted signal region we'll construct: from (last_window_start + step) up to last_window_start + step*out_len + L
    forecast_start = last_window_start + step
    forecast_end = last_window_start + step * out_len + L
    forecast_len = forecast_end - forecast_start
    predicted_acc = np.zeros(forecast_len)
    normalizer = np.zeros(forecast_len)
    first_feat_unscaled = preds_unscaled[0]
    first_complex = real_features_to_complex_vector(first_feat_unscaled)
    first_timewin = np.fft.irfft(first_complex, n=L)
    plt.figure(figsize=(10,4))
    plt.plot(np.arange(L)/sr, signal[-L:], label='Last observed window (time domain)')
    plt.plot(np.arange(L)/sr, first_timewin, label='First predicted window (one-step)')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.title('Last observed window vs first predicted window')
    plt.show()
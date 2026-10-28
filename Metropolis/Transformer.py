from time import time

from numpy import random

from keras.layers import LayerNormalization
from tensorflow import cast, float32, math, equal, reduce_sum, function, GradientTape, data
from tensorflow._api.v2 import train
# from tensorflow import data
from tensorflow.keras.backend import sparse_categorical_crossentropy, argmax
from tensorflow.keras.layers import Layer, Dense, ReLU, Dropout, Input
from tensorflow.keras.metrics import Mean
from tensorflow.keras.models import Model
# from tensorflow.python.keras.optimizer_v1 import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.optimizer_v2.learning_rate_schedule import LearningRateSchedule

from Metropolis.MultiHeadAttention import MultiHeadAttention
from Metropolis.PositionalEncoding import PositionEmbeddingFixedWeights

from pickle import load
from numpy.random import shuffle
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from tensorflow import convert_to_tensor, int64, maximum, linalg, ones


class PrepareDataset:
	def __init__(self, **kwargs):
		super(PrepareDataset, self).__init__(**kwargs)
		self.n_sentences = 10000  # Number of sentences to include in the dataset
		self.train_split = 0.9  # Ratio of the training data split

	# Fit a tokenizer
	def create_tokenizer(self, dataset):
		tokenizer = Tokenizer()
		tokenizer.fit_on_texts(dataset)

		return tokenizer

	def find_seq_length(self, dataset):
		return max(len(seq.split()) for seq in dataset)

	def find_vocab_size(self, tokenizer, dataset):
		tokenizer.fit_on_texts(dataset)

		return len(tokenizer.word_index) + 1

	def __call__(self, filename, **kwargs):
		# Load a clean dataset
		clean_dataset = load(open(filename, 'rb'))

		# Reduce dataset size
		dataset = clean_dataset[:self.n_sentences, :]

		# Include start and end of string tokens
		for i in range(dataset[:, 0].size):
			dataset[i, 0] = "<START> " + dataset[i, 0] + " <EOS>"
			dataset[i, 1] = "<START> " + dataset[i, 1] + " <EOS>"

		# Random shuffle the dataset
		shuffle(dataset)

		# Split the dataset
		train = dataset[:int(self.n_sentences * self.train_split)]

		# Prepare tokenizer for the encoder input
		enc_tokenizer = self.create_tokenizer(train[:, 0])
		enc_seq_length = self.find_seq_length(train[:, 0])
		enc_vocab_size = self.find_vocab_size(enc_tokenizer, train[:, 0])

		# Encode and pad the input sequences
		trainX = enc_tokenizer.texts_to_sequences(train[:, 0])
		trainX = pad_sequences(trainX, maxlen=enc_seq_length, padding='post')
		trainX = convert_to_tensor(trainX, dtype=int64)

		# Prepare tokenizer for the decoder input
		dec_tokenizer = self.create_tokenizer(train[:, 1])
		dec_seq_length = self.find_seq_length(train[:, 1])
		dec_vocab_size = self.find_vocab_size(dec_tokenizer, train[:, 1])

		# Encode and pad the input sequences
		trainY = dec_tokenizer.texts_to_sequences(train[:, 1])
		trainY = pad_sequences(trainY, maxlen=dec_seq_length, padding='post')
		trainY = convert_to_tensor(trainY, dtype=int64)

		return trainX, trainY, train, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size


class FeedForward(Layer):
    def __init__(self, d_ff, d_model, **kwargs):
        super(FeedForward, self).__init__(**kwargs)
        self.fullyConnected1 = Dense(d_ff)
        self.fullyConnected2 = Dense(d_model)
        self.activation = ReLU()

    def call(self, x):
        fullyConnected1 = self.fullyConnected1(x)
        return self.fullyConnected2(self.activation(fullyConnected1))


class AddNormalization(Layer):
    def __init__(self, **kwargs):
        super(AddNormalization, self).__init__(**kwargs)
        self.layerNorm = LayerNormalization()

    def call(self, x, sublayerX):
        add = x + sublayerX
        return self.layerNorm(add)


class EncoderLayer(Layer):
    def __init__(self, sequence_length, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.build(input_shape=[None, sequence_length, d_model])
        self.d_model = d_model
        self.sequence_length = sequence_length
        self.multiHeadAttention = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.add_norm1 = AddNormalization()
        self.feed_forward = FeedForward(d_ff, d_model)
        self.dropout2 = Dropout(rate)
        self.add_norm2 = AddNormalization()

    def build_graph(self):
        input_layer = Input(shape=(self.sequence_length, self.d_model))
        return Model(inputs=[input_layer], outputs=self.call(input_layer, None, True))

    def call(self, x, paddingMask, training):
        multiHeadOupout = self.multiHeadAttention(x, x, x, paddingMask)
        multiHeadOupout = self.dropout1(multiHeadOupout, training=training)

        addNormOutput = self.addNorm1(x, multiHeadOupout)

        feedForwardOutput = self.feedForward(addNormOutput)
        feedForwardOutput = self.dropout2(feedForwardOutput, training=training)

        return self.addNorm2(addNormOutput, feedForwardOutput)


class Encoder(Layer):
    def __init__(self, vocabSize, seqLength, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.positionEncoding = PositionEmbeddingFixedWeights(seqLength, vocabSize, d_model)
        self.dropout = Dropout(rate)
        self.encoderLayer = [EncoderLayer(seqLength, h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, inputSequence, paddingMask, training):
        positionEncodingOutput = self.positionEncoding(inputSequence)

        x = self.dropout(positionEncodingOutput, training=training)

        for i, layer in enumerate(self.encoderLayer):
            x = layer(x, paddingMask, training)

        return x


class DecoderLayer(Layer):
    def __init__(self, h, d_k, d_v, d_model, d_ff, rate, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)
        self.multiHeadAttention1 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout1 = Dropout(rate)
        self.addNorm1 = AddNormalization()
        self.multiHeadAttention2 = MultiHeadAttention(h, d_k, d_v, d_model)
        self.dropout2 = Dropout(rate)
        self.addNorm2 = AddNormalization()
        self.feedForward = FeedForward(d_ff, d_model)
        self.dropout3 = Dropout(rate)
        self.addNorm3 = AddNormalization()

    def call(self, x, encoderOutput, lookaheadMask, paddingMask, training):
        multiHeadOutput1 = self.multiHeadAttention1(x, x, x, lookaheadMask)
        multiHeadOutput1 = self.dropout1(multiHeadOutput1, training=training)

        addNormOutput1 = self.addNorm1(x, multiHeadOutput1)

        multiHeadOutput2 = self.multiHeadAttention2(addNormOutput1, encoderOutput, encoderOutput, paddingMask)
        multiHeadOutput2 = self.dropout1(multiHeadOutput2, training=training)

        addNormOutput2 = self.addNorm2(addNormOutput1, multiHeadOutput2)

        feedForwardOutput = self.feedForward(addNormOutput2)
        feedForwardOutput = self.dropout3(feedForwardOutput)

        return self.addNorm3(addNormOutput2, feedForwardOutput)


class Decoder(Layer):
    def __init__(self, vocabSize, seqLength, h, d_k, d_v, d_model, d_ff, n, rate, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.posEncoding = PositionEmbeddingFixedWeights(seqLength, vocabSize, d_model)
        self.dropout = Dropout(rate)
        self.decoderLayer = [DecoderLayer(h, d_k, d_v, d_model, d_ff, rate) for _ in range(n)]

    def call(self, outputTarget, encoderOutput, lookaheadMask, paddingMask, training):
        posEncodingOutput = self.posEncoding(outputTarget)
        x = self.dropout(posEncodingOutput, training=training)

        for i, layer in enumerate(self.decoderLayer):
            x = layer(x, encoderOutput, lookaheadMask, paddingMask, training)

        return x


class TransformerModel(Model):
    def __init__(self, encoderVocabSize, decoderVocabSize, encoderSeqLength, decoderSeqLength, h, d_k, d_v, d_model, d_ff_inner, n, rate, **kwargs):
        super(TransformerModel, self).__init__(**kwargs)
        self.encoder = Encoder(encoderVocabSize, encoderSeqLength, h, d_k, d_v, d_model, d_ff_inner, n, rate)
        self.decoder = Decoder(decoderVocabSize, decoderSeqLength, h, d_k, d_v, d_model, d_ff_inner, n, rate)
        self.modelLastLayer = Dense(decoderVocabSize)

    def paddingMask(self, input):
        # Create mask which marks the zero padding values in the input by a 1
        mask = math.equal(input, 0)
        mask = cast(mask, float32)

        return mask

    def lookaheadMask(self, shape):
        # Mask out future entries by marking them with a 1.0
        mask = 1 - linalg.band_part(ones((shape, shape)), -1, 0)

        return mask

    def call(self, encoderInput, decoderInput, training):
        encoderPaddingMask = self.paddingMask(encoderInput)

        decoderInPaddingMask = self.paddingMask(decoderInput)
        decoderInLookaheadMask = self.lookaheadMask(decoderInput.shape[1])
        decoderInLookaheadMask = maximum(decoderInPaddingMask, decoderInLookaheadMask)

        encoderOutput = self.encoder(encoderInput, encoderPaddingMask, training)
        decoderOutput = self.decoder(decoderInput, encoderOutput, decoderInLookaheadMask, encoderPaddingMask, training)

        return self.modelLastLayer(decoderOutput)


class LRScheduler(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, **kwargs):
        super(LRScheduler, self).__init__(**kwargs)

        self.d_model = cast(d_model, float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step_num):
        # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
        arg1 = step_num ** -0.5
        arg2 = step_num * (self.warmup_steps ** -1.5)

        return (self.d_model ** -0.5) * math.minimum(arg1, arg2)


def loss_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of loss
    padding_mask = math.logical_not(equal(target, 0))
    padding_mask = cast(padding_mask, float32)

    # Compute a sparse categorical cross-entropy loss on the unmasked values
    loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

    # Compute the mean loss over the unmasked values
    return reduce_sum(loss) / reduce_sum(padding_mask)


# Defining the accuracy function
def accuracy_fcn(target, prediction):
    # Create mask so that the zero padding values are not included in the computation of accuracy
    padding_mask = math.logical_not(equal(target, 0))

    # Find equal prediction and target values, and apply the padding mask
    accuracy = equal(target, argmax(prediction, axis=2))
    accuracy = math.logical_and(padding_mask, accuracy)

    # Cast the True/False values to 32-bit-precision floating-point numbers
    padding_mask = cast(padding_mask, float32)
    accuracy = cast(accuracy, float32)

    # Compute the mean accuracy over the unmasked values
    return reduce_sum(accuracy) / reduce_sum(padding_mask)

# Speeding up the training process
@function
def train_step(encoder_input, decoder_input, decoder_output):
    with GradientTape() as tape:
        # Run the forward pass of the model to generate a prediction
        prediction = training_model(encoder_input, decoder_input, training=True)

        # Compute the training loss
        loss = loss_fcn(decoder_output, prediction)

        # Compute the training accuracy
        accuracy = accuracy_fcn(decoder_output, prediction)

    # Retrieve gradients of the trainable variables with respect to the training loss
    gradients = tape.gradient(loss, training_model.trainable_weights)

    # Update the values of the trainable variables by gradient descent
    optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

    train_loss(loss)
    train_accuracy(accuracy)


if __name__ == "__main__":
    enc_vocab_size = 20  # Vocabulary size for the encoder
    dec_vocab_size = 20  # Vocabulary size for the decoder

    enc_seq_length = 5  # Maximum length of the input sequence
    dec_seq_length = 5  # Maximum length of the target sequence
    # Define the model parameters
    h = 8  # Number of self-attention heads
    d_k = 64  # Dimensionality of the linearly projected queries and keys
    d_v = 64  # Dimensionality of the linearly projected values
    d_model = 512  # Dimensionality of model layers' outputs
    d_ff = 2048  # Dimensionality of the inner fully connected layer
    n = 6  # Number of layers in the encoder stack

    # Define the training parameters
    epochs = 2
    batch_size = 64
    beta_1 = 0.9
    beta_2 = 0.98
    epsilon = 1e-9
    dropout_rate = 0.1

    encoder = EncoderLayer(enc_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
    encoder.build_graph().summary()

    decoder = DecoderLayer(dec_seq_length, h, d_k, d_v, d_model, d_ff, dropout_rate)
    decoder.build_graph().summary()

    # Implementing a learning rate scheduler
    class LRScheduler(LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000, **kwargs):
            super(LRScheduler, self).__init__(**kwargs)

            self.d_model = cast(d_model, float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step_num):
            # Linearly increasing the learning rate for the first warmup_steps, and decreasing it thereafter
            arg1 = step_num ** -0.5
            arg2 = step_num * (self.warmup_steps ** -1.5)

            return (self.d_model ** -0.5) * math.minimum(arg1, arg2)


    # Instantiate an Adam optimizer
    optimizer = Adam(LRScheduler(d_model), beta_1, beta_2, epsilon)

    # Prepare the training and test splits of the dataset
    dataset = PrepareDataset()
    trainX, trainY, train_orig, enc_seq_length, dec_seq_length, enc_vocab_size, dec_vocab_size = dataset(
        'english-german-both.pkl')

    # Prepare the dataset batches
    train_dataset = data.Dataset.from_tensor_slices((trainX, trainY))
    train_dataset = train_dataset.batch(batch_size)

    # Create model
    training_model = TransformerModel(enc_vocab_size, dec_vocab_size, enc_seq_length, dec_seq_length, h, d_k, d_v,
                                      d_model, d_ff, n, dropout_rate)


    # Defining the loss function
    def loss_fcn(target, prediction):
        # Create mask so that the zero padding values are not included in the computation of loss
        padding_mask = math.logical_not(equal(target, 0))
        padding_mask = cast(padding_mask, float32)

        # Compute a sparse categorical cross-entropy loss on the unmasked values
        loss = sparse_categorical_crossentropy(target, prediction, from_logits=True) * padding_mask

        # Compute the mean loss over the unmasked values
        return reduce_sum(loss) / reduce_sum(padding_mask)


    # Defining the accuracy function
    def accuracy_fcn(target, prediction):
        # Create mask so that the zero padding values are not included in the computation of accuracy
        padding_mask = math.logical_not(equal(target, 0))

        # Find equal prediction and target values, and apply the padding mask
        accuracy = equal(target, argmax(prediction, axis=2))
        accuracy = math.logical_and(padding_mask, accuracy)

        # Cast the True/False values to 32-bit-precision floating-point numbers
        padding_mask = cast(padding_mask, float32)
        accuracy = cast(accuracy, float32)

        # Compute the mean accuracy over the unmasked values
        return reduce_sum(accuracy) / reduce_sum(padding_mask)


    # Include metrics monitoring
    train_loss = Mean(name='train_loss')
    train_accuracy = Mean(name='train_accuracy')

    # Create a checkpoint object and manager to manage multiple checkpoints
    ckpt = train.Checkpoint(model=training_model, optimizer=optimizer)
    ckpt_manager = train.CheckpointManager(ckpt, "./checkpoints", max_to_keep=3)


    # Speeding up the training process
    @function
    def train_step(encoder_input, decoder_input, decoder_output):
        with GradientTape() as tape:
            # Run the forward pass of the model to generate a prediction
            prediction = training_model(encoder_input, decoder_input, training=True)

            # Compute the training loss
            loss = loss_fcn(decoder_output, prediction)

            # Compute the training accuracy
            accuracy = accuracy_fcn(decoder_output, prediction)

        # Retrieve gradients of the trainable variables with respect to the training loss
        gradients = tape.gradient(loss, training_model.trainable_weights)

        # Update the values of the trainable variables by gradient descent
        optimizer.apply_gradients(zip(gradients, training_model.trainable_weights))

        train_loss(loss)
        train_accuracy(accuracy)


    for epoch in range(epochs):

        train_loss.reset_states()
        train_accuracy.reset_states()

        print("\nStart of epoch %d" % (epoch + 1))

        start_time = time()

        # Iterate over the dataset batches
        for step, (train_batchX, train_batchY) in enumerate(train_dataset):

            # Define the encoder and decoder inputs, and the decoder output
            encoder_input = train_batchX[:, 1:]
            decoder_input = train_batchY[:, :-1]
            decoder_output = train_batchY[:, 1:]

            train_step(encoder_input, decoder_input, decoder_output)

            if step % 50 == 0:
                print(
                    f'Epoch {epoch + 1} Step {step} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')
                # print("Samples so far: %s" % ((step + 1) * batch_size))

        # Print epoch number and loss value at the end of every epoch
        print("Epoch %d: Training Loss %.4f, Training Accuracy %.4f" % (
        epoch + 1, train_loss.result(), train_accuracy.result()))

        # Save a checkpoint after every five epochs
        if (epoch + 1) % 5 == 0:
            save_path = ckpt_manager.save()
            print("Saved checkpoint at epoch %d" % (epoch + 1))

    print("Total time taken: %.2fs" % (time() - start_time))
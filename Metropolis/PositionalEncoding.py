from keras.layers import TextVectorization
from tensorflow.python.keras.layers import Embedding, Layer
import tensorflow as tf
from tensorflow import convert_to_tensor, string
from tensorflow.data import Dataset
import numpy as np
import matplotlib.pyplot as plt


class PositionEmbeddingLayer(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingLayer, self).__init__(**kwargs)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


class PositionEmbeddingFixedWeights(Layer):
    def __init__(self, sequence_length, vocab_size, output_dim, **kwargs):
        super(PositionEmbeddingFixedWeights, self).__init__(**kwargs)
        word_embedding_matrix = self.get_position_encoding(vocab_size, output_dim)
        position_embedding_matrix = self.get_position_encoding(sequence_length, output_dim)
        self.word_embedding_layer = Embedding(
            input_dim=vocab_size, output_dim=output_dim,
            weights=[word_embedding_matrix],
            trainable=False
        )
        self.position_embedding_layer = Embedding(
            input_dim=sequence_length, output_dim=output_dim,
            weights=[position_embedding_matrix],
            trainable=False
        )

    def get_position_encoding(self, seq_len, d, n=10000):
        P = np.zeros((seq_len, d))
        for k in range(seq_len):
            for i in np.arange(int(d / 2)):
                denominator = np.power(n, 2 * i / d)
                P[k, 2 * i] = np.sin(k / denominator)
                P[k, 2 * i + 1] = np.cos(k / denominator)
        return P

    def call(self, inputs):
        position_indices = tf.range(tf.shape(inputs)[-1])
        embedded_words = self.word_embedding_layer(inputs)
        embedded_indices = self.position_embedding_layer(position_indices)
        return embedded_words + embedded_indices


if __name__ == "__main__":
    output_sequence_length = 5
    vocab_size = 10
    sentences = [["I am a robot"], ["you too robot"]]
    sentence_data = Dataset.from_tensor_slices(sentences)
    # Create the TextVectorization layer
    vectorize_layer = TextVectorization(
        output_sequence_length=output_sequence_length,
        max_tokens=vocab_size)
    # Train the layer to create a dictionary
    vectorize_layer.adapt(sentence_data)
    # Convert all sentences to tensors
    word_tensors = convert_to_tensor(sentences, dtype=tf.string)
    # Use the word tensors to get vectorized phrases
    vectorized_words = vectorize_layer(word_tensors)
    print("Vocabulary: ", vectorize_layer.get_vocabulary())
    print("Vectorized words: ", vectorized_words)

    output_length = 6
    word_embedding_layer = Embedding(vocab_size, output_length)
    embedded_words = word_embedding_layer(vectorized_words)
    print(embedded_words)

    position_embedding_layer = Embedding(output_sequence_length, output_length)
    position_indices = tf.range(output_sequence_length)
    embedded_indices = position_embedding_layer(position_indices)
    print(embedded_indices)

    final_output_embedding = embedded_words + embedded_indices
    print("Final output: ", final_output_embedding)
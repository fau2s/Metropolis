{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {
    "collapsed": false
   },
   "outputs": [],
   "source": [
    "from keras.datasets import cifar10,mnist\n",
    "from keras.preprocessing.image import ImageDataGenerator \n",
    "from keras.models import Sequential, Model\n",
    "from keras.layers import Dense, Input, Dropout, Activation, Flatten, LSTM, SimpleRNN, GRU, Embedding\n",
    "from keras.layers import Convolution2D, MaxPooling2D, normalization\n",
    "from keras.utils import np_utils\n",
    "from keras.preprocessing import sequence\n",
    "\n",
    "import keras.backend as K"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://s3.amazonaws.com/text-datasets/imdb.npz\n"
     ]
    }
   ],
   "source": [
    "from keras.datasets import imdb\n",
    "\n",
    "(x_train, y_train), (x_test, y_test) = imdb.load_data(path=\"imdb_full.pkl\",\n",
    "                                                      num_words=None,\n",
    "                                                      skip_top=0,\n",
    "                                                      maxlen=None,\n",
    "                                                      seed=113,\n",
    "                                                      start_char=1,\n",
    "                                                      oov_char=2,\n",
    "                                                      index_from=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "from keras.datasets import reuters\n",
    "\n",
    "(x_train, y_train), (x_test, y_test) = reuters.load_data(path=\"reuters.pkl\",\n",
    "                                                         num_words=None,\n",
    "                                                         skip_top=0,\n",
    "                                                         maxlen=None,\n",
    "                                                         test_split=0.2,\n",
    "                                                         seed=113,\n",
    "                                                         start_char=1,\n",
    "                                                         oov_char=2,\n",
    "                                                         index_from=3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Downloading data from https://s3.amazonaws.com/keras-datasets/boston_housing.npz\n"
     ]
    }
   ],
   "source": [
    "from keras.datasets import boston_housing\n",
    "\n",
    "(x_train, y_train), (x_test, y_test) = boston_housing.load_data()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "top_words = 5000\n",
    "(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "max_review_length = 500\n",
    "X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)\n",
    "X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "sigmoid_cross_entropy_with_logits() got an unexpected keyword argument 'labels'",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-26-2b4e0bacfd91>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      4\u001b[0m \u001b[0;31m#model.add(LSTM(100))\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m \u001b[0;31m#model.add(Dense(1, activation='sigmoid'))\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 6\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompile\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'binary_crossentropy'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0moptimizer\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m'adam'\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmetrics\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'accuracy'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      7\u001b[0m \u001b[0;32mprint\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msummary\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/models.pyc\u001b[0m in \u001b[0;36mcompile\u001b[0;34m(self, optimizer, loss, metrics, sample_weight_mode, **kwargs)\u001b[0m\n\u001b[1;32m    764\u001b[0m                            \u001b[0mmetrics\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmetrics\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    765\u001b[0m                            \u001b[0msample_weight_mode\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msample_weight_mode\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 766\u001b[0;31m                            **kwargs)\n\u001b[0m\u001b[1;32m    767\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptimizer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moptimizer\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    768\u001b[0m         \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloss\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/engine/training.pyc\u001b[0m in \u001b[0;36mcompile\u001b[0;34m(self, optimizer, loss, metrics, loss_weights, sample_weight_mode, **kwargs)\u001b[0m\n\u001b[1;32m    897\u001b[0m             \u001b[0mloss_weight\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mloss_weights_list\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mi\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    898\u001b[0m             output_loss = weighted_loss(y_true, y_pred,\n\u001b[0;32m--> 899\u001b[0;31m                                         sample_weight, mask)\n\u001b[0m\u001b[1;32m    900\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0mlen\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;34m>\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    901\u001b[0m                 \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmetrics_tensors\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_loss\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/engine/training.pyc\u001b[0m in \u001b[0;36mweighted\u001b[0;34m(y_true, y_pred, weights, mask)\u001b[0m\n\u001b[1;32m    428\u001b[0m         \"\"\"\n\u001b[1;32m    429\u001b[0m         \u001b[0;31m# score_array has ndim >= 2\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 430\u001b[0;31m         \u001b[0mscore_array\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    431\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mmask\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    432\u001b[0m             \u001b[0;31m# Cast the mask to floatX to avoid float64 upcasting in theano\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/losses.pyc\u001b[0m in \u001b[0;36mbinary_crossentropy\u001b[0;34m(y_true, y_pred)\u001b[0m\n\u001b[1;32m     43\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     44\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mbinary_crossentropy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 45\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbinary_crossentropy\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_true\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     46\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     47\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/backend/tensorflow_backend.pyc\u001b[0m in \u001b[0;36mbinary_crossentropy\u001b[0;34m(output, target, from_logits)\u001b[0m\n\u001b[1;32m   2613\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2614\u001b[0m     return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,\n\u001b[0;32m-> 2615\u001b[0;31m                                                    logits=output)\n\u001b[0m\u001b[1;32m   2616\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   2617\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: sigmoid_cross_entropy_with_logits() got an unexpected keyword argument 'labels'"
     ]
    }
   ],
   "source": [
    "embedding_vector_length = 32\n",
    "model = Sequential()\n",
    "model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))\n",
    "#model.add(LSTM(100))\n",
    "#model.add(Dense(1, activation='sigmoid'))\n",
    "model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "print(model.summary())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Loading data...\n",
      "(25000, 'train sequences')\n",
      "(25000, 'test sequences')\n",
      "Pad sequences (samples x time)\n",
      "('x_train shape:', (25000, 80))\n",
      "('x_test shape:', (25000, 80))\n"
     ]
    }
   ],
   "source": [
    "max_features = 20000\n",
    "maxlen = 80  \n",
    "batch_size = 32\n",
    "\n",
    "(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)\n",
    "print(len(x_train), 'train sequences')\n",
    "print(len(x_test), 'test sequences')\n",
    "\n",
    "print('Pad sequences (samples x time)')\n",
    "x_train = sequence.pad_sequences(x_train, maxlen=maxlen)\n",
    "x_test = sequence.pad_sequences(x_test, maxlen=maxlen)\n",
    "print('x_train shape:', x_train.shape)\n",
    "print('x_test shape:', x_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[   15,   256,     4, ...,    19,   178,    32],\n",
       "       [  125,    68,     2, ...,    16,   145,    95],\n",
       "       [  645,   662,     8, ...,     7,   129,   113],\n",
       "       ..., \n",
       "       [  529,   443, 17793, ...,     4,  3586,     2],\n",
       "       [  286,  1814,    23, ...,    12,     9,    23],\n",
       "       [   97,    90,    35, ...,   204,   131,     9]], dtype=int32)"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "x_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "ename": "TypeError",
     "evalue": "Expected int32, got list containing Tensors of type '_Message' instead.",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[0;32m<ipython-input-18-b506a68d298e>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mSequential\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mEmbedding\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmax_features\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m128\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mLSTM\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;36m128\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/models.pyc\u001b[0m in \u001b[0;36madd\u001b[0;34m(self, layer)\u001b[0m\n\u001b[1;32m    453\u001b[0m                           output_shapes=[self.outputs[0]._keras_shape])\n\u001b[1;32m    454\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 455\u001b[0;31m             \u001b[0moutput_tensor\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mlayer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0moutputs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    456\u001b[0m             \u001b[0;32mif\u001b[0m \u001b[0misinstance\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0moutput_tensor\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlist\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    457\u001b[0m                 raise TypeError('All layers in a Sequential model '\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/layers/recurrent.pyc\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, inputs, initial_state, **kwargs)\u001b[0m\n\u001b[1;32m    250\u001b[0m             \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    251\u001b[0m                 \u001b[0mkwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'initial_state'\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0minitial_state\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 252\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0msuper\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mRecurrent\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m__call__\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    253\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    254\u001b[0m     \u001b[0;32mdef\u001b[0m \u001b[0mcall\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmask\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minitial_state\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mNone\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtraining\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/engine/topology.pyc\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, inputs, **kwargs)\u001b[0m\n\u001b[1;32m    552\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    553\u001b[0m             \u001b[0;31m# Actually call the layer, collecting output(s), mask(s), and shape(s).\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 554\u001b[0;31m             \u001b[0moutput\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcall\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    555\u001b[0m             \u001b[0moutput_mask\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcompute_mask\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mprevious_mask\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    556\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/layers/recurrent.pyc\u001b[0m in \u001b[0;36mcall\u001b[0;34m(self, inputs, mask, initial_state, training)\u001b[0m\n\u001b[1;32m    288\u001b[0m                              'or `batch_shape` argument to your Input layer.')\n\u001b[1;32m    289\u001b[0m         \u001b[0mconstants\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_constants\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtraining\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 290\u001b[0;31m         \u001b[0mpreprocessed_input\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mpreprocess_input\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0minputs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtraining\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mNone\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    291\u001b[0m         last_output, outputs, states = K.rnn(self.step,\n\u001b[1;32m    292\u001b[0m                                              \u001b[0mpreprocessed_input\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/layers/recurrent.pyc\u001b[0m in \u001b[0;36mpreprocess_input\u001b[0;34m(self, inputs, training)\u001b[0m\n\u001b[1;32m   1031\u001b[0m                                           \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdropout\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0minput_dim\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0munits\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1032\u001b[0m                                           timesteps, training=training)\n\u001b[0;32m-> 1033\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcatenate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mx_i\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_f\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_c\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mx_o\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m2\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1034\u001b[0m         \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1035\u001b[0m             \u001b[0;32mreturn\u001b[0m \u001b[0minputs\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/keras/backend/tensorflow_backend.pyc\u001b[0m in \u001b[0;36mconcatenate\u001b[0;34m(tensors, axis)\u001b[0m\n\u001b[1;32m   1523\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msparse_concat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0maxis\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtensors\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1524\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1525\u001b[0;31m         \u001b[0;32mreturn\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconcat\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mto_dense\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mx\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0mx\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mtensors\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0maxis\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1526\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1527\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/ops/array_ops.pyc\u001b[0m in \u001b[0;36mconcat\u001b[0;34m(concat_dim, values, name)\u001b[0m\n\u001b[1;32m   1073\u001b[0m       ops.convert_to_tensor(concat_dim,\n\u001b[1;32m   1074\u001b[0m                             \u001b[0mname\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;34m\"concat_dim\"\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1075\u001b[0;31m                             \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mint32\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mget_shape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m   1076\u001b[0m                             ).assert_is_compatible_with(tensor_shape.scalar())\n\u001b[1;32m   1077\u001b[0m       \u001b[0;32mreturn\u001b[0m \u001b[0midentity\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;36m0\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mscope\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/framework/ops.pyc\u001b[0m in \u001b[0;36mconvert_to_tensor\u001b[0;34m(value, dtype, name, as_ref, preferred_dtype)\u001b[0m\n\u001b[1;32m    667\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    668\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mret\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 669\u001b[0;31m           \u001b[0mret\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mconversion_func\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalue\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mas_ref\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mas_ref\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    670\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    671\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mret\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0mNotImplemented\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/framework/constant_op.pyc\u001b[0m in \u001b[0;36m_constant_tensor_conversion_function\u001b[0;34m(v, dtype, name, as_ref)\u001b[0m\n\u001b[1;32m    174\u001b[0m                                          as_ref=False):\n\u001b[1;32m    175\u001b[0m   \u001b[0m_\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mas_ref\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 176\u001b[0;31m   \u001b[0;32mreturn\u001b[0m \u001b[0mconstant\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mv\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mname\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mname\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    177\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    178\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/framework/constant_op.pyc\u001b[0m in \u001b[0;36mconstant\u001b[0;34m(value, dtype, shape, name, verify_shape)\u001b[0m\n\u001b[1;32m    163\u001b[0m   \u001b[0mtensor_value\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mattr_value_pb2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mAttrValue\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    164\u001b[0m   tensor_value.tensor.CopyFrom(\n\u001b[0;32m--> 165\u001b[0;31m       tensor_util.make_tensor_proto(value, dtype=dtype, shape=shape, verify_shape=verify_shape))\n\u001b[0m\u001b[1;32m    166\u001b[0m   \u001b[0mdtype_value\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mattr_value_pb2\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mAttrValue\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mtensor_value\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtensor\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    167\u001b[0m   const_tensor = g.create_op(\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/framework/tensor_util.pyc\u001b[0m in \u001b[0;36mmake_tensor_proto\u001b[0;34m(values, dtype, shape, verify_shape)\u001b[0m\n\u001b[1;32m    365\u001b[0m       \u001b[0mnparray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mempty\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mshape\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp_dt\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    366\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 367\u001b[0;31m       \u001b[0m_AssertCompatible\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    368\u001b[0m       \u001b[0mnparray\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mnp\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0marray\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mvalues\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtype\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mnp_dt\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    369\u001b[0m       \u001b[0;31m# check to them.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;32m/Users/balint/.graphlab/anaconda/lib/python2.7/site-packages/tensorflow/python/framework/tensor_util.pyc\u001b[0m in \u001b[0;36m_AssertCompatible\u001b[0;34m(values, dtype)\u001b[0m\n\u001b[1;32m    300\u001b[0m     \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    301\u001b[0m       raise TypeError(\"Expected %s, got %s of type '%s' instead.\" %\n\u001b[0;32m--> 302\u001b[0;31m                       (dtype.name, repr(mismatch), type(mismatch).__name__))\n\u001b[0m\u001b[1;32m    303\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    304\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
      "\u001b[0;31mTypeError\u001b[0m: Expected int32, got list containing Tensors of type '_Message' instead."
     ]
    }
   ],
   "source": [
    "model = Sequential()\n",
    "model.add(Embedding(max_features, 128))\n",
    "model.add(LSTM(128))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Embedding(max_features, 128))\n",
    "model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))\n",
    "\n",
    "model.add(Dense(1, activation='sigmoid'))\n",
    "\n",
    "# try using different optimizers and different optimizer configs\n",
    "model.compile(loss='binary_crossentropy',\n",
    "              optimizer='adam',\n",
    "              metrics=['accuracy'])\n",
    "\n",
    "print('Train...')\n",
    "model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=15,\n",
    "          validation_data=(x_test, y_test))\n",
    "score, acc = model.evaluate(x_test, y_test,\n",
    "                            batch_size=batch_size)\n",
    "print('Test score:', score)\n",
    "print('Test accuracy:', acc)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "# as the first layer in a Sequential model\n",
    "model = Sequential()\n",
    "model.add(LSTM(32, input_shape=(10, 64)))\n",
    "# now model.output_shape == (None, 32)\n",
    "# note: `None` is the batch dimension.\n",
    "\n",
    "# for subsequent layers, no need to specify the input size:\n",
    "model.add(LSTM(16))\n",
    "\n",
    "# to stack recurrent layers, you must use return_sequences=True\n",
    "# on any recurrent layer that feeds into another recurrent layer.\n",
    "# note that you only need to specify the input size on the first layer.\n",
    "model = Sequential()\n",
    "model.add(LSTM(64, input_dim=64, input_length=10, return_sequences=True))\n",
    "model.add(LSTM(32, return_sequences=True))\n",
    "model.add(LSTM(10))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": [
    "model = Sequential()\n",
    "model.add(Embedding(1000, 64, input_length=10))\n",
    "# the model will take as input an integer matrix of size (batch, input_length).\n",
    "# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).\n",
    "# now model.output_shape == (None, 10, 64), where None is the batch dimension.\n",
    "\n",
    "input_array = np.random.randint(1000, size=(32, 10))\n",
    "\n",
    "model.compile('rmsprop', 'mse')\n",
    "output_array = model.predict(input_array)\n",
    "assert output_array.shape == (32, 10, 64)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Not using data augmentation.\n",
      "Train on 50000 samples, validate on 10000 samples\n",
      "Epoch 1/10\n",
      " 1120/50000 [..............................] - ETA: 408s - loss: 2.3128 - acc: 0.0902"
     ]
    }
   ],
   "source": [
    "if not data_augmentation: \n",
    "    print('Not using data augmentation.')    \n",
    "    model.fit(X_train, Y_train,  batch_size=batch_size, nb_epoch=nb_epoch,\n",
    "              validation_data=(X_test, Y_test),  shuffle=True)\n",
    "else:\n",
    "    print('Using real-time data augmentation.')    \n",
    "    datagen = ImageDataGenerator( \n",
    "        featurewise_center=False,  # set input mean to 0 over the dataset \n",
    "        samplewise_center=False,  # set each sample mean to 0 \n",
    "        featurewise_std_normalization=False,  # divide inputs by std of the dataset \n",
    "        samplewise_std_normalization=False,  # divide each input by its std \n",
    "        zca_whitening=False,  # apply ZCA whitening        \n",
    "        rotation_range=0,  # randomly rotate images in the range (0..180 deg)        \n",
    "        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)        \n",
    "        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)        \n",
    "        horizontal_flip=True,  # randomly flip images \n",
    "        vertical_flip=False  # randomly flip images    \n",
    "        )\n",
    "    datagen.fit(X_train)    # Fit the model on the batches generated by datagen.flow().\n",
    "    model.fit_generator(datagen.flow(X_train, Y_train, \n",
    "                                     batch_size=batch_size), samples_per_epoch=X_train.shape[0],\n",
    "                                     nb_epoch=nb_epoch, validation_data=(X_test, Y_test))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "collapsed": true
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "anaconda-cloud": {},
  "kernelspec": {
   "display_name": "Python 2",
   "language": "python",
   "name": "python2"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 2
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython2",
   "version": "2.7.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 0
}
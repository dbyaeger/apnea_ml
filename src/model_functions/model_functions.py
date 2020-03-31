#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 15:02:09 2020

@author: danielyaeger
"""

from tensorflow import keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Conv1D, Dropout, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import MaxPool1D
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

######### Functions for building network #####################################
def _add_conv_layer(layer, n_filters, filter_size, l2_lambda = 0,
                    max_pool=True, stride=1, input_shape=None, activation="relu", 
                    batch_norm=True):
    """
    Add a conv layer to network architecture
    INPUTS:
        layer: input to layer
        n_filters: nunber of filters in current layer
        filter_size: filter size for current layer
        l2_lambda: l2 regularizer value
        max_pool: will current layer use max pooling
        stride: stride for current layer
        input_shape: input shape to current layer, only needed for first layer
        activation: activation function for current layer
        batch_norm: will current layer use batch normalization
        conv2d: is current layer going to use 2d conv (True) or 1d (False)
    OUTPUTS:
        conv layer
    """
    if input_shape:
        layer = Conv1D(n_filters, filter_size, kernel_regularizer = l2(l2_lambda),
                     strides=stride, input_shape=input_shape)(layer)
    else:
        layer = Conv1D(n_filters, filter_size, strides=stride, 
                     kernel_regularizer = l2(l2_lambda))(layer)
    layer = Activation(activation)(layer)
    if batch_norm:
        layer = BatchNormalization()(layer)
    if max_pool:
        layer = MaxPool1D(2)(layer)
    return layer

def _add_lstm_layer(layer, units = 128, return_sequences = False, 
              dropout_p = 0, input_shape = None):
    """
    Add an LSTM layer to network architecture
    INPUTS:
        layer = input to layer
        units = number of units
        input_shape = input shape to current layer, only needed for first layer
        return_sequences = whether to return a sequence
        dropout_p = dropout probability
    OUTPUTs:
        LSTM layer
    """
    if input_shape:
        layer = LSTM(input_shape=input_shape, units = units, 
                     return_sequences = return_sequences, dropout = dropout_p,
                     activation = 'sigmoid')(layer)
    else:
        layer = LSTM(units = units, 
                     return_sequences = return_sequences, dropout = dropout_p,
                     activation = 'sigmoid')(layer)
    return layer


def _add_dense_layer(layer, n_out, l2_lambda = None, dropout_p=None,
                     activation = "relu"):
    """
    Add a dense layer to network architecture
    INPUTS:
        layer: input to layer
        n_out: number of output neurons
        activation: activation function for current layer
        dropout_p: retain probability for current layer (if None -> no dropout)
    OUTPUTS:
        dense layer
    """
    if l2_lambda:
        layer = Dense(n_out, kernel_regularizer = l2(l2_lambda))(layer)
    else:
        layer = Dense(n_out)(layer)
    if activation:
        layer = Activation(activation)(layer)
    if dropout_p:
        layer = Dropout(dropout_p)(layer)
    return layer

def build_model(**params):
    """
    Function for build a network based on the specified input params. By default,
    convolutional layers are assumed, when present, to come before lstm_layers.
    
    INPUTS:
        params = {"conv_layers":[(input_size_i, output_size_i, l2_lambda_i, max_pool), ...],
                  "fc_layers": [output_size_i, l2_lambda_i, dropout_i],
                  "input_shape":(n, m, ...),
                  "callbacks":[callback_i, ...],
                  "lstm_layers" ;[(units_i, return_sequences_i, dropout_p_i), ...],
                  "learning_rate":1e-3,
                  ...}
    OUTPUTS:
        compiled network
    """
    if "conv_layers" in params:
        conv_layers = params["conv_layers"]
    
    if "fc_layers" in params:
        fc_layers = params["fc_layers"]
    
    if "lstm_layers" in params:
        lstm_layers = params["lstm_layers"]
        
    input_ = Input(shape=params["input_shape"])
    if params.get("learning_rate"):
        learning_rate = params["learning_rate"]
    else:
        learning_rate = 1e-3
    out = input_
    if "conv_layers" in params:
        for i, p in enumerate(conv_layers):
            out = _add_conv_layer(out, *p)
        out = Flatten()(out)
    if "lstm_layers" in params:
        for i, p in enumerate(lstm_layers):
            out = _add_lstm_layer(out, *p)
    for i, p in enumerate(fc_layers):
        if i < len(fc_layers) - 1:
            out = _add_dense_layer(out, *p)
        else:
            n = p[0]
            out = _add_dense_layer(out, n, activation="softmax")
    optim = Adam(learning_rate)
    model = Model(input_, out)
    model.compile(optimizer=optim, 
                  loss="categorical_crossentropy",
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model

#def build_model_TPU(**params):
#    """
#    Function for build a network based on the specified input params. By default,
#    convolutional layers are assumed, when present, to come before lstm_layers.
#    
#    INPUTS:
#        params = {"conv_layers":[(input_size_i, output_size_i, l2_lambda_i), ...],
#                  "fc_layers": [output_size_i, l2_lambda_i, dropout_i],
#                  "input_shape":(n, m, ...),
#                  "callbacks":[callback_i, ...],
#                  "lstm_layers" ;[(units_i, return_sequences_i, dropout_p_i), ...],
#                  "learning_rate":1e-3,
#                  ...}
#    OUTPUTS:
#        compiled network
#    """
#    # clear tensorflow session
#    clear_session()
#    
#    if "conv_layers" in params:
#        conv_layers = params["conv_layers"]
#    
#    if "fc_layers" in params:
#        fc_layers = params["fc_layers"]
#    
#    if "lstm_layers" in params:
#        lstm_layers = params["lstm_layers"]
#        
#    input_ = Input(shape=params["input_shape"])
#    if params.get("learning_rate"):
#        learning_rate = params["learning_rate"]
#    else:
#        learning_rate = 1e-3
#    out = input_
#    if "conv_layers" in params:
#        for i, p in enumerate(conv_layers):
#            out = _add_conv_layer(out, *p)
#        out = Flatten()(out)
#    if "lstm_layers" in params:
#        for i, p in enumerate(lstm_layers):
#            out = _add_lstm_layer(out, *p)
#    for i, p in enumerate(fc_layers):
#        if i < len(fc_layers) - 1:
#            out = _add_dense_layer(out, *p)
#        else:
#            n = p[0]
#            out = _add_dense_layer(out, n, activation="softmax")
#    model = Model(input_, out)
#    
#    # Convert model to tpu model and compile with tensorflow optimizer
#    tpu_model = tf.contrib.tpu.keras_to_tpu_model(model,
#                strategy= tf.contrib.tpu.TPUDistributionStrategy(
#                        tf.contrib.cluster_resolver.TPUClusterResolver(
#                        tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])))
#    
#    tpu_model.compile(optimizer=tf.train.AdamOptimizer(earning_rate=learning_rate), 
#                  loss=categorical_crossentropy,
#                  metrics=['categorial_accuracy'])
#    return model
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 16:19:46 2020

@author: danielyaeger
"""

import tensorflow as tf

def process(x,y):
    """Takes in a numpy array, processes it and converts it to appropriate
    format for TRFRecord
    """
    
    feature = {
      'label': _int64_feature(y),
      'data': _bytes_feature(tf.io.serialize_tensor(x)),
      }

  return tf.train.Example(features=tf.train.Features(feature=feature))
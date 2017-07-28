#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

class TextCNNPredictor(object):

    def __init__(self,checkpoint_dir="",batch_size=64,allow_soft_placement=True,log_device_placement=False):
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement

    def predict(self,texts):
    
        # Map data into vocabulary
        vocab_path = os.path.join(self.checkpoint_dir, "..", "vocab")
        vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        x_test = np.array(list(vocab_processor.transform(texts)))
    
        print("\nEvaluating...\n")
    
        # Evaluation
        # ==================================================
        checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto(
              allow_soft_placement=self.allow_soft_placement,
              log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
        
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
        
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                confidences = graph.get_operation_by_name("output/confidences").outputs[0]
        
                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x_test), self.batch_size, 1, shuffle=False)
        
                # Collect the predictions here
                all_predictions = []
                all_confidences = []
        
                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                    batch_confidences = sess.run(confidences, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_confidences.extend(batch_confidences)
    
        return (all_predictions,all_confidences)












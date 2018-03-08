import tensorflow as tf
import numpy as np
import os
from helpers import prefix_to_file

# Once model has been frozen and saved locally as .pb, use this script to store it to S3
# TODO: make this process streamlined
if __name__ == "__main__":
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}/model/test_tf_model.meta'.format(prefix_to_file(__file__)))
        saver.restore(sess, tf.train.latest_checkpoint('{}/model/'.format(prefix_to_file(__file__))))

        # Ensure basic graph was imported without any glaring issues (i.e. it can still run inference)
        x_placeholder = tf.placeholder(tf.float32, [None, 3, 200, 100], name='input')
        x = np.zeros((3, 200, 100))

        y_out = sess.run('dense_2/Softmax:0', feed_dict={'conv2d_1_input:0': [x]})

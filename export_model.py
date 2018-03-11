import tensorflow as tf
import numpy as np
import os, argparse
import boto3
from datetime import datetime
from helpers import prefix_to_file

BUCKET_NAME = 'codetroopa-impostor'
dir = os.path.dirname(os.path.realpath(__file__))

# A simpler approach to the Tensorflow freeze_graph.py script
# From: https://gist.github.com/morgangiraud/249505f540a5e53a48b0c1a869d370bf#file-medium-tffreeze-1-py
def freeze_graph(model_dir, output_node_names):
    if not tf.gfile.Exists(model_dir):
        raise AssertionError(
            "Export directory doesn't exists. Please specify an export "
            "directory: %s" % model_dir)

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path

    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    output_graph_path = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    # We start a session using a temporary fresh Graph
    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constants
        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess, # The session is used to retrieve the weights
            tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes
            output_node_names.split(",") # The output node names are used to select the usefull nodes
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph_path, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))

    return output_graph_path


# Once model has been frozen and saved locally as .pb, use this script to store it to S3
# TODO: make this process streamlined
if __name__ == "__main__":
    # Args for exporting model
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model folder to export")
    parser.add_argument("--output_node_names", type=str, default="", help="The name of the output nodes, comma separated.")
    args = parser.parse_args()

    file_path = freeze_graph(args.model_dir, args.output_node_names)

    # After successfully freezing the graph, we store it on S3
    client = boto3.client('s3')
    with open(file_path, 'rb') as f:
        client.put_object(
            Bucket=BUCKET_NAME,
            Key='models/model_{}.pb'.format(datetime.now().strftime('%Y%m%d-%H%M')),
            Body=f
        )
    print('Successfully stored model on S3')

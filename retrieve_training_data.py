import numpy as np
import csv
import boto3
import json
import pickle
from datetime import datetime
from helpers import prefix_to_file


BUCKET_NAME = 'codetroopa-impostor'
TSET_PREFIX = 'unprocessed_training_data/'

PLAYER_KEY = 'playerMatrices.csv'
OBSTACLE_KEY = 'obstacleMatrices.csv'
ENEMY_KEY = 'enemyMatrices.csv'
YLABEL_KEY = 'ylabels.csv'
META_KEY = 'metadata.json'

def folder_names():
    response = client.list_objects_v2(
        Bucket=BUCKET_NAME,
        Prefix=TSET_PREFIX,
        Delimiter='/'
    )
    return [x['Prefix'] for x in response['CommonPrefixes']]

def reshape_data(pmatrix, omatrix, ematrix, ylabels):
    # reshape 3 matrices so we can concatenate them together
    pmatrix = pmatrix.reshape(1, pmatrix.shape[0], pmatrix.shape[1], pmatrix.shape[2])
    omatrix = omatrix.reshape(1, omatrix.shape[0], omatrix.shape[1], omatrix.shape[2])
    ematrix = ematrix.reshape(1, ematrix.shape[0], ematrix.shape[1], ematrix.shape[2])

    # concatenate the three matrices together, then ensure that the 0th axis represents the number of training samples
    x_data = np.concatenate((pmatrix, omatrix, ematrix)).swapaxes(0, 1)

    # Assert that we have the same number of input data as labelled data
    assert(x_data.shape[0] == ylabels.shape[0])
    return (x_data, ylabels)


# This returns all three matrices needed for training
def get_matrices_from_s3():
    player_matrices = None
    obstacle_matrices = None
    enemy_matrices = None
    y_labels = None

    folder_keys = folder_names()

    # go through each folder, appending objects to the corresponding matrices
    for folder_prefix in folder_keys:
        print('Retrieving objects from folder: {}'.format(folder_prefix))
        # First, we grab meta-data specific to this folder
        response = client.get_object(Bucket=BUCKET_NAME, Key=folder_prefix + META_KEY)
        meta_data = json.loads(response['Body'].read().decode('utf-8'))
        m_count = meta_data['matrix_count']
        m_columns = meta_data['matrix_length']
        m_rows = meta_data['matrix_width']
        y_length = meta_data['ylabel_length']

        # init matrices TODO: somehow refactor this out
        if player_matrices is None:
            player_matrices = np.ndarray((0, m_columns, m_rows))
        if obstacle_matrices is None:
            obstacle_matrices = np.ndarray((0, m_columns, m_rows))
        if enemy_matrices is None:
            enemy_matrices = np.ndarray((0, m_columns, m_rows))
        if y_labels is None:
            y_labels = np.ndarray((0, y_length))

        # Now we go through each type of matrix, reshaping into an appropriate format
        response = client.get_object(Bucket=BUCKET_NAME, Key=folder_prefix + PLAYER_KEY)
        player_matrix = np.fromstring(response['Body'].read().decode('utf-8'), sep='\n', dtype=int)
        player_matrices = np.concatenate((player_matrices, player_matrix.reshape((m_count, m_columns, m_rows))))

        response = client.get_object(Bucket=BUCKET_NAME, Key=folder_prefix + OBSTACLE_KEY)
        obstacle_matrix = np.fromstring(response['Body'].read().decode('utf-8'), sep='\n', dtype=int)
        obstacle_matrices = np.concatenate((obstacle_matrices, obstacle_matrix.reshape((m_count, m_columns, m_rows))))

        response = client.get_object(Bucket=BUCKET_NAME, Key=folder_prefix + ENEMY_KEY)
        enemy_matrix = np.fromstring(response['Body'].read().decode('utf-8'), sep='\n', dtype=int)
        enemy_matrices = np.concatenate((enemy_matrices, enemy_matrix.reshape((m_count, m_columns, m_rows))))

        # Finally, add the labelled data
        response = client.get_object(Bucket=BUCKET_NAME, Key=folder_prefix + YLABEL_KEY)
        y_label = np.fromstring(response['Body'].read().decode('utf-8'), sep='\n', dtype=int)
        y_labels = np.concatenate((y_labels, y_label.reshape((m_count, y_length))))

    assert(player_matrices.shape[0] == obstacle_matrices.shape[0])
    assert(player_matrices.shape[0] == enemy_matrices.shape[0])
    return (player_matrices, obstacle_matrices, enemy_matrices, y_labels)

if __name__ == '__main__':
    client = boto3.client('s3')

    # First, we go through all objects in our training_sets folder
    (player_matrices, obstacle_matrices, enemy_matrices, y_labels) = get_matrices_from_s3()

    # Reshape training data into a collection of 3D matrix
    (x_data, y_labels) = reshape_data(player_matrices, obstacle_matrices, enemy_matrices, y_labels)

    # Serialize training data locally
    data = { 'xlabels': x_data, 'ylabels': y_labels }
    with open('{}/training_set.pkl'.format(prefix_to_file(__file__)), 'wb') as f:
        pickle.dump(data, f)

    # Store training set in AWS S3 for future use / backup
    with open('{}/training_set.pkl'.format(prefix_to_file(__file__)), 'rb') as f:
        client.put_object(
            Bucket=BUCKET_NAME,
            Key='training_sets/training_set_{}.pkl'.format(datetime.now().strftime('%Y%m%d-%H%M')),
            Body=f
        )

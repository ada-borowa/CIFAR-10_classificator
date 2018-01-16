"""
CNN codes extraction.
"""
import copy
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
from data_processing import load_data, create_image
CLASSES = open('labels.txt', 'r').read().split('\n')

INCEPTION_GRAPH = '../models/inception_v3_graph.pb'
DATA_PATH = '../cifar-10_data/'


def setup_graph(filename):
    """ Creates graph from *.pb file."""
    with open(filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='inception')


def extract_cnn_codes(data):
    """Extracts cnn codes from pool3 layer."""
    codes = []
    images = data[b'data']
    labels = data[b'labels']
    len_data = len(images)

    with tf.Session() as session:
        layer = session.graph.get_tensor_by_name('inception/pool_3:0')
        for i, img in enumerate(images):
            if i % 100 == 0:
                print('Processing {}/{}'.format(i, len_data))
            img = create_image(img)
            prediction = session.run(layer, {'inception/DecodeJpeg:0': img})
            codes.append(np.squeeze(prediction))

    return codes, labels


def change_dimension(codes, labels):
    """Changes dimension of set using t-SNE to prepare for visualization.
    And reduces number of data points to save time."""
    n = 2500
    clf = manifold.TSNE(n_components=2, init='pca', random_state=0)
    codes = copy.deepcopy(codes)
    codes = np.array(codes)
    idx = np.random.choice(codes.shape[0], n, replace=False)
    X = clf.fit_transform(codes[idx])
    labels = np.array(labels)
    return X, labels[idx]


def visualize_cnn_codes(X, labels):
    """Visualizes CNN codes preprocessed with t-SNE."""
    fig = plt.figure()
    for c in set(labels):
        x = np.array([X[i] for i in range(len(X)) if labels[i] == c])
        plt.scatter(x[:, 0], x[:, 1], alpha=0.6, label=CLASSES[c])
    plt.legend(bbox_to_anchor=(0., 1., 1., 0.), loc=3, ncol=5,
               mode="expand", borderaxespad=0.)
    plt.show()

if __name__ == '__main__':
    # load data and setup tf graph
    all_data = load_data()
    setup_graph(INCEPTION_GRAPH)

    # extract codes from pool3 layer and pickle them
    train_cnn_codes, train_labels = extract_cnn_codes(all_data['train'])
    pickle.dump(train_cnn_codes, open(DATA_PATH + 'train_cnn_codes', 'wb'))
    pickle.dump(train_labels, open(DATA_PATH + 'train_labels', 'wb'))

    test_cnn_codes, test_labels = extract_cnn_codes(all_data['test'])
    pickle.dump(test_cnn_codes, open(DATA_PATH + 'test_cnn_codes', 'wb'))
    pickle.dump(test_labels, open(DATA_PATH + 'test_labels', 'wb'))

    # load codes from pickles
    train_codes = pickle.load(open(DATA_PATH + 'train_cnn_codes', 'rb'))
    train_labels = pickle.load(open(DATA_PATH + 'train_labels', 'rb'))

    # use t-SNE on training data and visualize part of them
    X, X_labels = change_dimension(train_codes, train_labels)
    visualize_cnn_codes(X, X_labels)

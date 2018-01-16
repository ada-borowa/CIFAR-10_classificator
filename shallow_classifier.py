"""Training shallow classifier: Random Forest."""
import numpy as np
import tensorflow as tf
from skimage.feature import hog
from skimage import color

from data_processing import create_image, load_data


def get_hog_features(images):
    """Gets HOG vectors of the list of images."""
    features = []
    for i, img in enumerate(images):
        if (i + 1) % 10000 == 0:
            print('Processing {}/{}'.format(i + 1, len(images)))
        img = color.rgb2gray(create_image(img))
        fd, _ = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualise=True)
        features.append(fd)
    return np.array(features)


def random_forest(train_data, test_data):
    """Trains and tests random forest classificator."""
    train_x, train_y = train_data
    test_x, test_y = test_data
    params = tf.contrib.tensor_forest.python.tensor_forest.ForestHParams(
        num_classes=10, num_features=len(train_x[0]), regression=False,
        num_trees=50, max_nodes=1000)
    clf = tf.contrib.tensor_forest.client.random_forest.TensorForestEstimator(params)
    clf.fit(x=np.array(train_x, dtype='float32'), y=train_y)
    predictions = clf.predict(x=np.array(test_x, dtype='float32'))
    predictions = np.array(list([p['classes'] for p in list(predictions)]))
    return sum(sum([predictions == test_y])) / len(test_y)

if __name__ == '__main__':
    data = load_data()

    train_images = data['train'][b'data']
    train_labels = data['train'][b'labels']
    train_features = get_hog_features(train_images)
    test_images = data['test'][b'data']
    test_labels = data['test'][b'labels']
    test_features = get_hog_features(test_images)

    accuracy = random_forest((train_features, train_labels), (test_features, test_labels))
    print('Accuracy of random forest: {}'.format(accuracy))  # 0.4357

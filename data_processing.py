"""
Data processing and visualization functions.
"""
import os
import pickle
import numpy as np
import requests
import tarfile
import matplotlib.pyplot as plt


SOURCE = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
DATA_PATH = '../cifar-10_data/'
BATCH_PATH = DATA_PATH + 'cifar-10-batches-py/'
DATA_FILENAME = SOURCE.split('/')[-1]
CLASSES = open('labels.txt', 'r').read().split('\n')


def download_data():
    """
    Downloads data if needed, and then extracts them to given location.
    """
    if not os.path.exists(DATA_PATH + DATA_FILENAME):
        request = requests.get(SOURCE, stream=True)
        if request.status_code == 200:
            with open(DATA_PATH + DATA_FILENAME, 'wb') as f:
                print('Downloading...')
                for chunk in request.iter_content():
                    if chunk:
                        f.write(chunk)
            print("File downloaded to {}/{}.".format(DATA_PATH, DATA_FILENAME))

        else:
            print("Error: couldn't connect.")
    else:
        print("File exists.")

    print('Extracting...')
    tar = tarfile.open(DATA_PATH + DATA_FILENAME)
    tar.extractall(path=DATA_PATH)
    tar.close()


def load_data():
    """
    Unpickles data from batches (both training and testing) and returns a dictionary with all the data. 
    """
    files = ['data_batch_{}'.format(i) for i in range(1, 6)]
    data_dict = {'train': {b'data': [], b'labels': [], b'filenames': []}, 'test': {}}
    for f in files:
        with open(BATCH_PATH + f, 'rb') as f_pickle:
            new_batch = pickle.load(f_pickle, encoding='bytes')
            del new_batch[b'batch_label']
            for key in data_dict['train']:
                data_dict['train'][key].extend(new_batch[key])
    with open(BATCH_PATH + 'test_batch', 'rb') as f_pickle:
        data_dict['test'] = pickle.load(f_pickle, encoding='bytes')
    del data_dict['test'][b'batch_label']
    return data_dict


def get_image(data_dict, n):
    """
    Returns n-th image from training set.
    """
    return data_dict['train'][b'data'][n]


def create_image(array):
    """
    An image is given by 1x3072 array, it's first reshaped to create a list of pixels, and then once more to create
    a RGB image.
    """
    array = np.transpose(np.reshape(array, (3, 1024)))
    return np.reshape(array, (32, 32, 3))


def plot_data(data_dict):
    """
    Plots 10 random images from each category. 
    """
    size = (10, 11)
    data_size = len(data_dict['train'][b'data'])
    fig = plt.figure()
    for i in range(size[0]):
        category = CLASSES[i]
        fig.add_subplot(size[0], size[1], i*size[1] + 1)
        plt.text(-1, 0.3, category)
        plt.axis('off')
        indeces = [x for x in range(0, data_size) if data_dict['train'][b'labels'][x] == i]
        random_images = np.random.choice(indeces, 10)
        for j, r_img in enumerate(random_images):
            img = create_image(get_image(data_dict, r_img))
            fig.add_subplot(size[0], size[1], i * size[1] + j + 2)
            plt.imshow(img)
            plt.axis('off')
    plt.show()


if __name__ == '__main__':
    download_data()
    data = load_data()
    plot_data(data)

"""
Training SVM classifier.
"""
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


DATA_PATH = '../cifar-10_data/'
train_codes = np.array(pickle.load(open(DATA_PATH + 'train_cnn_codes', 'rb')))
train_labels = np.array(pickle.load(open(DATA_PATH + 'train_labels', 'rb')))
test_codes = np.array(pickle.load(open(DATA_PATH + 'test_cnn_codes', 'rb')))
test_labels = np.array(pickle.load(open(DATA_PATH + 'test_labels', 'rb')))

if __name__ == '__main__':
    # Using only 10000 images, because of time
    idx = np.random.choice([i for i in range(len(train_codes))], 10000)
    X_train = train_codes[idx, :]
    y_train = train_labels[idx]

    # kernels = ['linear', 'rbf']
    # Cs = [1, 10, 100, 1000]
    # gammas = [0.1, 0.01, 0.001, 0.0001]

    svm_params = [{'kernel': ['linear'],
                   'C': [1, 10, 100, 1000]},
                  {'kernel': ['rbf'],
                   'C': [1, 10, 100, 1000],
                   'gamma': [0.1, 0.01, 0.001, 0.0001]}]

    svm = SVC()

    clf = GridSearchCV(svm, svm_params, cv=5, n_jobs=4, verbose=3)
    clf.fit(X_train, y_train)
    print(clf.best_params_)  # {'gamma': 0.001, 'C': 10, 'kernel': 'rbf'}

    best_C = float(clf.best_params_['C'])
    clf2 = None
    if clf.best_params_['kernel'] == 'linear':
        clf2 = SVC(kernel='linear', C=best_C)
    else:
        best_gamma = float(clf.best_params_['gamma'])
        clf2 = SVC(kernel='rbf', C=best_C, gamma=best_gamma)
    clf2.fit(train_codes, train_labels)

    # just in case
    svm_file = open('svm1', 'wb')
    pickle.dump(clf2, svm_file)
    svm_file.close()

    print(clf2.score(test_codes, test_labels))  # 0.9104


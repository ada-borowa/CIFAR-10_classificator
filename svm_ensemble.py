import pickle
import numpy as np
import time
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


DATA_PATH = '../cifar-10_data/'
train_codes = np.array(pickle.load(open(DATA_PATH + 'train_cnn_codes', 'rb')))
train_labels = np.array(pickle.load(open(DATA_PATH + 'train_labels', 'rb')))
test_codes = np.array(pickle.load(open(DATA_PATH + 'test_cnn_codes', 'rb')))
test_labels = np.array(pickle.load(open(DATA_PATH + 'test_labels', 'rb')))

X_train, X_test, y_train, y_test = train_test_split(train_codes, train_labels, test_size=0.2)


def bagging_result(kernel, C, gamma, max_samples, max_features):
    bagging = BaggingClassifier(SVC(kernel=kernel, C=C, gamma=gamma),
                                max_samples=max_samples,
                                max_features=max_features,
                                n_jobs=4,
                                verbose=3)
    time0 = time.time()
    bagging.fit(X_train, y_train)
    print('Fitted in: {} s.'.format(time.time() - time0))
    score = bagging.score(X_test, y_test)
    print('Score: {}'.format(score))
    return score


if __name__ == '__main__':

    max_result = -1
    max_params = {'C': -1.0, 'gamma': -1.0}
    for C in [1.0, 10.0, 100.0, 1000.0]:
        for gamma in [0.1, 0.01, 0.001, 0.0001]:
            print(C, gamma)
            score = bagging_result('rbf', C, gamma, 0.5, 0.5)
            if score > max_result:
                max_result = score
                max_params['C'] = C
                max_params['gamma'] = gamma

    print('The best result: C={}, gamma={}, score={}'.format(max_params['C'], max_params['gamma'], max_result))
    # The best result: C=100.0, gamma=0.001, score=0.9053
    for n in [0.5, 1.0]:
        bagging = BaggingClassifier(SVC(kernel='rbf', C=max_params['C'], gamma=max_params['gamma']),
                                    max_samples=n,
                                    max_features=n,
                                    n_jobs=4,
                                    verbose=3)
        time0 = time.time()
        bagging.fit(train_codes, train_labels)
        print('max_samples={}, max_features={}'.format(n, n))
        print('Fitted in: {} s.'.format(time.time() - time0))
        score = bagging.score(test_codes, test_labels)
        print('Score: {}'.format(score))  # 0.9033

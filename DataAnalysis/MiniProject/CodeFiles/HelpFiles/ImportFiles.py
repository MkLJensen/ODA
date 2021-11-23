from sklearn.model_selection import train_test_split
from MiniProject.CodeFiles.HelpFiles import MyPCA
import scipy.io
import numpy as np


def import_mnist(pca=False):
    mat = scipy.io.loadmat('../DataFiles/mnist_loaded.mat')
    test_images = np.transpose(mat['test_images'])
    test_labels = mat['test_labels'].ravel()
    train_images = np.transpose(mat['train_images'])
    train_labels = mat['train_labels'].ravel()

    if pca:
        train_images, test_images = MyPCA.do_PCA(2, train_images, test_images)

    return [test_images, test_labels, train_images, train_labels]


def import_orl(pca=False):
    orl_labels = scipy.io.loadmat('../DataFiles/orl_lbls.mat')['lbls'].ravel()
    orl_data = np.transpose(scipy.io.loadmat('../DataFiles/orl_data.mat')['data'])

    train_data, test_data, train_labels, test_labels = train_test_split(orl_data, orl_labels,
                                                                        test_size=0.2, random_state=41)
    if pca:
        train_data, test_data = MyPCA.do_PCA(2, train_data, test_data)

    return [train_data, test_data, train_labels, test_labels]
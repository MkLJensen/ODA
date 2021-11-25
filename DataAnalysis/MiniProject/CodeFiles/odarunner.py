import MiniProject.CodeFiles.project_perceptron_backpropagation as pcb
import MiniProject.CodeFiles.project_perceptron_mse as pcm
import MiniProject.CodeFiles.project_nearest_neighbor as nnc
import MiniProject.CodeFiles.project_nearest_class_centroid as ncc
import MiniProject.CodeFiles.project_nearest_subclass_centroid as nsc
import matplotlib.pyplot as plt
import numpy as np


def plot_double_bar(orldata, mnistdata, N, xlabel, ylabel, title, xticks, width=0.35):
    ind = np.arange(N)
    plt.bar(ind, np.array(list(orldata.items()))[:, 1].astype(float), width, label='ORL')
    plt.bar(ind + width, np.array(list(mnistdata.items()))[:, 1].astype(float), width, label='MNIST')
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.xticks(ind + width / 2, xticks)
    plt.legend(loc='best')
    plt.show()


def run_perceptron_backpropagation():
    perceptron_backpropagation_pca = {}
    perceptron_backpropagation_no_pca = {}

    learning_rate = 0.003
    perceptron_backpropagation_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=True)
    perceptron_backpropagation_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=True)
    perceptron_backpropagation_no_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=False)
    perceptron_backpropagation_no_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=False)

    learning_rate = 0.03
    perceptron_backpropagation_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=True)
    perceptron_backpropagation_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=True)
    perceptron_backpropagation_no_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=False)
    perceptron_backpropagation_no_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=False)

    learning_rate = 0.3
    perceptron_backpropagation_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=True)
    perceptron_backpropagation_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=True)
    perceptron_backpropagation_no_pca["orl_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_orl(learning_rate, pca=False)
    perceptron_backpropagation_no_pca["mnist_" + str(learning_rate)] = \
        pcb.perceptron_backpropagation_mnist(learning_rate, pca=False)

    filtered_orl_pca = dict(filter(lambda item: "orl" in item[0], perceptron_backpropagation_pca.items()))
    filtered_orl_no_pca = dict(filter(lambda item: "orl" in item[0], perceptron_backpropagation_no_pca.items()))
    filtered_mnist_pca = dict(filter(lambda item: "mnist" in item[0], perceptron_backpropagation_pca.items()))
    filtered_mnist_no_pca = dict(filter(lambda item: "mnist" in item[0], perceptron_backpropagation_no_pca.items()))

    plot_double_bar(filtered_orl_pca, filtered_mnist_pca, 3, 'Learning Rate', 'Accuracy',
                    "Accuracy by LR and Dataset with PCA (Perceptron Backpropagation)", ('0.003', '0.03', '0.3'))

    plot_double_bar(filtered_orl_no_pca, filtered_mnist_no_pca, 3, 'Learning Rate', 'Accuracy',
                    "Accuracy by LR and Dataset (Perceptron Backpropagation)", ('0.003', '0.03', '0.3'))


def run_perceptron_mse():
    learning_rate = 0.003
    perceptron_mse_pca = {}
    perceptron_mse_no_pca = {}

    perceptron_mse_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=True)
    perceptron_mse_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=True)
    perceptron_mse_no_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=False)
    perceptron_mse_no_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=False)

    learning_rate = 0.03
    perceptron_mse_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=True)
    perceptron_mse_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=True)
    perceptron_mse_no_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=False)
    perceptron_mse_no_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=False)

    learning_rate = 0.3
    perceptron_mse_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=True)
    perceptron_mse_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=True)
    perceptron_mse_no_pca["orl_" + str(learning_rate)] = \
        pcm.perceptron_mse_orl(learning_rate, pca=False)
    perceptron_mse_no_pca["mnist_" + str(learning_rate)] = \
        pcm.perceptron_mse_mnist(learning_rate, pca=False)

    filtered_orl_pca = dict(filter(lambda item: "orl" in item[0], perceptron_mse_pca.items()))
    filtered_orl_no_pca = dict(filter(lambda item: "orl" in item[0], perceptron_mse_no_pca.items()))
    filtered_mnist_pca = dict(filter(lambda item: "mnist" in item[0], perceptron_mse_pca.items()))
    filtered_mnist_no_pca = dict(filter(lambda item: "mnist" in item[0], perceptron_mse_no_pca.items()))

    plot_double_bar(filtered_orl_pca, filtered_mnist_pca, 3, 'Learning Rate', 'Accuracy',
                    "Accuracy by LR and Dataset with PCA (Perceptron MSE)", ('0.003', '0.03', '0.3'))

    plot_double_bar(filtered_orl_no_pca, filtered_mnist_no_pca, 3, 'Learning Rate', 'Accuracy',
                    "Accuracy by LR and Dataset (Perceptron MSE)", ('0.003', '0.03', '0.3'))


def run_subclass_centroid():
    subclass_centroid_pca = {}
    subclass_centroid_no_pca = {}

    subclass_centroid_pca["orl_" + str(2)] = \
        nsc.nearest_subclass_centroid_orl(2, pca=True)
    subclass_centroid_pca["mnist_" + str(2)] = \
        nsc.nearest_subclass_centroid_mnist(2, pca=True)
    subclass_centroid_no_pca["orl_" + str(2)] = \
        nsc.nearest_subclass_centroid_orl(2, pca=False)
    subclass_centroid_no_pca["mnist_" + str(2)] = \
        nsc.nearest_subclass_centroid_mnist(2, pca=False)

    subclass_centroid_pca["orl_" + str(3)] = \
        nsc.nearest_subclass_centroid_orl(3, pca=True)
    subclass_centroid_pca["mnist_" + str(3)] = \
        nsc.nearest_subclass_centroid_mnist(3, pca=True)
    subclass_centroid_no_pca["orl_" + str(3)] = \
        nsc.nearest_subclass_centroid_orl(3, pca=False)
    subclass_centroid_no_pca["mnist_" + str(3)] = \
        nsc.nearest_subclass_centroid_mnist(3, pca=False)

    subclass_centroid_pca["orl_" + str(5)] = \
        nsc.nearest_subclass_centroid_orl(5, pca=True)
    subclass_centroid_pca["mnist_" + str(5)] = \
        nsc.nearest_subclass_centroid_mnist(5, pca=True)
    subclass_centroid_no_pca["orl_" + str(5)] = \
        nsc.nearest_subclass_centroid_orl(5, pca=False)
    subclass_centroid_no_pca["mnist_" + str(5)] = \
        nsc.nearest_subclass_centroid_mnist(5, pca=False)

    filtered_orl_pca = dict(filter(lambda item: "orl" in item[0], subclass_centroid_pca.items()))
    filtered_orl_no_pca = dict(filter(lambda item: "orl" in item[0], subclass_centroid_no_pca.items()))
    filtered_mnist_pca = dict(filter(lambda item: "mnist" in item[0], subclass_centroid_pca.items()))
    filtered_mnist_no_pca = dict(filter(lambda item: "mnist" in item[0], subclass_centroid_no_pca.items()))

    plot_double_bar(filtered_orl_pca, filtered_mnist_pca, 3, 'Subclasses', 'Accuracy',
                    "Accuracy by Subclasses and Dataset with PCA (Subclass Centroid)", ('2', '3', '5'))

    plot_double_bar(filtered_orl_no_pca, filtered_mnist_no_pca, 3, 'Learning Rate', 'Accuracy',
                    "Accuracy by Subclasses and Dataset with no PCA (Subclass Centroid)", ('2', '3', '5'))


def run_class_centroid():
    class_centroid_pca = {}
    class_centroid_no_pca = {}

    class_centroid_pca["orl"] = \
        ncc.nearest_centroid_orl(pca=True)
    class_centroid_pca["mnist"] = \
        ncc.nearest_centroid_mnist(pca=True)
    class_centroid_no_pca["orl"] = \
        ncc.nearest_centroid_orl(pca=False)
    class_centroid_no_pca["mnist"] = \
        ncc.nearest_centroid_mnist(pca=False)

    ind = np.arange(2)

    plt.bar(ind, np.array(list(class_centroid_pca.items()))[:, 1].astype(float), 0.35, label='PCA')
    plt.bar(ind + 0.35, np.array(list(class_centroid_no_pca.items()))[:, 1].astype(float), 0.35, label='No PCA')
    plt.ylabel('Accuracy')
    plt.xlabel('Dataset')
    plt.title('Accuracy and Dataset (Nearest Class Centroid)')
    plt.xticks(ind + 0.35 / 2, ('ORL', 'MNIST'))
    plt.legend(loc='best')
    plt.show()


def run_nearest_neighbor():
    nearest_neighbor_pca = {}
    nearest_neighbor_no_pca = {}

    nearest_neighbor_pca["orl_" + str(2)] = \
        nnc.nearest_neighbor_orl(2, pca=True)
    nearest_neighbor_pca["mnist_" + str(2)] = \
        nnc.nearest_neighbor_mnist(2, pca=True)
    nearest_neighbor_no_pca["orl_" + str(2)] = \
        nnc.nearest_neighbor_orl(2, pca=False)
    nearest_neighbor_no_pca["mnist_" + str(2)] = \
        nnc.nearest_neighbor_mnist(2, pca=False)

    nearest_neighbor_pca["orl_" + str(3)] = \
        nnc.nearest_neighbor_orl(3, pca=True)
    nearest_neighbor_pca["mnist_" + str(3)] = \
        nnc.nearest_neighbor_mnist(3, pca=True)
    nearest_neighbor_no_pca["orl_" + str(3)] = \
        nnc.nearest_neighbor_orl(3, pca=False)
    nearest_neighbor_no_pca["mnist_" + str(3)] = \
        nnc.nearest_neighbor_mnist(3, pca=False)

    nearest_neighbor_pca["orl_" + str(5)] = \
        nnc.nearest_neighbor_orl(5, pca=True)
    nearest_neighbor_pca["mnist_" + str(5)] = \
        nnc.nearest_neighbor_mnist(5, pca=True)
    nearest_neighbor_no_pca["orl_" + str(5)] = \
        nnc.nearest_neighbor_orl(5, pca=False)
    nearest_neighbor_no_pca["mnist_" + str(5)] = \
        nnc.nearest_neighbor_mnist(5, pca=False)

    filtered_orl_pca = dict(filter(lambda item: "orl" in item[0], nearest_neighbor_pca.items()))
    filtered_orl_no_pca = dict(filter(lambda item: "orl" in item[0], nearest_neighbor_no_pca.items()))
    filtered_mnist_pca = dict(filter(lambda item: "mnist" in item[0], nearest_neighbor_pca.items()))
    filtered_mnist_no_pca = dict(filter(lambda item: "mnist" in item[0], nearest_neighbor_no_pca.items()))

    plot_double_bar(filtered_orl_pca, filtered_mnist_pca, 3, 'Nr. Neighbors', 'Accuracy',
                    "Accuracy by Nr. Neigbors and Dataset with PCA (Nearest Neighbor)", ('2', '3', '5'))

    plot_double_bar(filtered_orl_no_pca, filtered_mnist_no_pca, 3, 'Nr. Neighbors', 'Accuracy',
                    "Accuracy by Nr. Neighbors and Dataset with No PCA (Nearest Neighbor)", ('2', '3', '5'))


if __name__ == '__main__':
    run_nearest_neighbor()
    # run_subclass_centroid()
    # run_perceptron_backpropagation()
    # run_perceptron_mse()

    # run_class_centroid()



import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from matplotlib.colors import ListedColormap, rgb_to_hsv, hsv_to_rgb
import seaborn as sns
import scipy.ndimage as ndi


def showOrlPlot(images, labels):
    plt.imshow(ndi.rotate(images[0].reshape(30, 40), 270), cmap='gray')
    plt.title("ORL Image With Label" + str(labels[0]))
    plt.show()


def showMnistPlot(images, labels):
    plt.imshow(np.flip(ndi.rotate(images[2].reshape(28, -28), 270), axis=1), cmap='gray')
    plt.title("MNIST Image With Label" + str(labels[2]))
    plt.show()


def plotConfusionMatrixFromEstimator(x_test, y_test, labels, estimator, name, estimator_name, PCA, hyper=""):
    ConfusionMatrixDisplay.from_estimator(estimator, x_test, y_test, labels=labels)
    plt.title("Confusion Matrix for: " + str(name) + " With " + str(estimator_name) + " " + PCA + " " + hyper,
              fontsize=10)
    plt.plot()
    plt.show()


def plotConfusionMatrixFromPreds(y_pred, y_true, labels, name, estimator_name, PCA, hyper=""):
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels)
    plt.title("Confusion Matrix for: " + str(name) + " With " + str(estimator_name) + " " + PCA + " " + hyper,
              fontsize=10)
    plt.plot()
    plt.show()


class HelpPlots:
    """ Colors and DefColors are unused """
    def __init__(self):
        self.mnist_colors = []
        self.mnist_colors_light = []
        self.orl_colors = []
        self.orl_colors_light = []
        self.defColors()

    def defColors(self):
        self.mnist_colors = [list(np.random.choice(range(256), size=3)) for i in range(10)]
        self.orl_colors = [list(np.random.choice(range(256), size=3)) for i in range(40)]

        orl_colors_hsv = [rgb_to_hsv(i) for i in self.orl_colors]
        orl_colors_hsv_light = [np.array([i[0], i[1]*0.90, i[2]]) for i in orl_colors_hsv]
        self.orl_colors_light = [hsv_to_rgb(i) for i in orl_colors_hsv_light]

        mnist_colors_hsv = [rgb_to_hsv(i) for i in self.mnist_colors]
        mnist_colors_hsv_light = [np.array([i[0], i[1] * 0.90, i[2]]) for i in mnist_colors_hsv]
        self.mnist_colors_light = [hsv_to_rgb(i) for i in mnist_colors_hsv_light]

        self.mnist_colors_light = [np.array([i[0] / 256, i[1] / 256, i[2] / 256]) for i in self.mnist_colors_light]
        self.mnist_colors = [np.array([i[0] / 256, i[1] / 256, i[2] / 256]) for i in self.mnist_colors]

        self.orl_colors_light = [np.array([i[0] / 256, i[1] / 256, i[2] / 256]) for i in self.orl_colors_light]
        self.orl_colors = [np.array([i[0] / 256, i[1] / 256, i[2] / 256]) for i in self.orl_colors]

        self.mnist_colors = ListedColormap(self.mnist_colors)
        self.mnist_colors_light = ListedColormap(self.mnist_colors_light)

        self.orl_colors = ListedColormap(self.orl_colors)
        self.orl_colors_light = ListedColormap(self.orl_colors_light)

    """
    https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html
    """
    def plotScatterAndDecisionBoundaryOfClassifier(self, model, x_test, labels, name, number_of_class, mname, hyper=""):
        plt.rcParams.update({'font.size': 28})
        h = 0.02 # Step Size
        x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
        y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        #Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        #Z = Z.reshape(xx.shape)
        plt.figure(figsize=(32, 24))

        color_map = sns.color_palette("hls", len(np.unique(labels)), as_cmap=False)

        # Plot also the training points
        if name == "ORL":
            #plt.contour(xx, yy, Z, color_map='black')
            sns.scatterplot(x=x_test[:, 0], y=x_test[:, 1], hue=labels, palette=color_map, alpha=0.5, s=200,
                            edgecolor="black")
        elif name == "MNIST":
            #plt.contour(xx, yy, Z, color_map='black')
            sns.scatterplot(x=x_test[:, 0], y=x_test[:, 1], hue=labels, palette=color_map, alpha=0.5, s=100,
                            edgecolor="black")

        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(
            "%s-Class classification For %s using %s : %s" % (number_of_class, name, mname, hyper)
        )
        plt.xlabel("PCA DataLabel 1")
        plt.ylabel("PCA DataLabel 2")

        plt.rcParams.update(plt.rcParamsDefault)
        plt.plot()
        plt.show()

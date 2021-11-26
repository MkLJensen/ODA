from sklearn.metrics import classification_report, accuracy_score
from MiniProject.CodeFiles.HelpFiles import ImportFiles
from sklearn.linear_model import SGDClassifier
import MiniProject.CodeFiles.HelpFiles.HelpPlots as HelpP

alpha = 0
loss = 'hinge'
estimator_name = "Perceptron Backpropagation"


def perceptron_backpropagation_mnist(learning_rate, pca=False):
    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    sgdc = SGDClassifier(loss=loss, alpha=alpha, learning_rate='constant', eta0=learning_rate)

    sgdc.fit(train_images, train_labels)

    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    PCA_string = "PCA" if pca else "No PCA"

    pred = sgdc.predict(test_images)

    if pca:
        Hp = HelpP.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(sgdc, test_images, pred, "MNIST", 10, "Perceptron_Backp",
                                                      hyper=str(learning_rate))

    HelpP.plotConfusionMatrixFromEstimator(test_images, test_labels,
                                           [int(i) for i in lbls_names], sgdc, "MNIST",
                                           estimator_name, PCA_string, hyper="LR: " + str(learning_rate))

    # print(classification_report(test_labels, pred, target_names=lbls_names))
    return accuracy_score(test_labels, pred)


def perceptron_backpropagation_orl(learning_rate, pca=False):
    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    sgdc = SGDClassifier(loss=loss, alpha=alpha, learning_rate='constant', eta0=learning_rate)

    sgdc.fit(train_data, train_labels)

    lbls_names = []
    for i in range(40):
        lbls_names.append(str(i))

    PCA_string = "PCA" if pca else "No PCA"

    pred = sgdc.predict(test_data)

    if pca:
        Hp = HelpP.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(sgdc, test_data, pred, "ORL", 40, "Perceptron_Backp",
                                                      hyper=str(learning_rate))

    HelpP.plotConfusionMatrixFromEstimator(test_data, test_labels,
                                           [int(i) for i in lbls_names], sgdc, "ORL",
                                           estimator_name, PCA_string, hyper="LR: " + str(learning_rate))

    # print(classification_report(test_labels, pred, zero_division=0))
    return accuracy_score(test_labels, pred)


if __name__ == '__main__':
    perceptron_backpropagation_orl(0.003, pca=True)
    perceptron_backpropagation_mnist(0.003, pca=True)

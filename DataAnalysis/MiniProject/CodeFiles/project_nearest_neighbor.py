from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from MiniProject.CodeFiles.HelpFiles import ImportFiles
import HelpFiles.HelpPlots

estimator_name = "Nearest Neighbor"


def nearest_neighbor_mnist(n_neighbors, pca=False):

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_images, train_labels)

    pred = knn.predict(test_images)
    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    if pca:
        Hp = HelpFiles.HelpPlots.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(knn, test_images, test_labels, "MNIST", 10)

    HelpFiles.HelpPlots.plotConfusionMatrixFromEstimator(test_images, test_labels, [int(i) for i in lbls_names],
                                                         knn, "MNIST", estimator_name)

    #print(classification_report(test_labels, pred, target_names=lbls_names))
    return accuracy_score(test_labels, pred)


def nearest_neighbor_orl(n_neighbors, pca=False):

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_data, train_labels)

    pred = knn.predict(test_data)

    lbls_names = []
    for i in range(40):
        lbls_names.append(str(i))

    if pca:
        Hp = HelpFiles.HelpPlots.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(knn, test_data, test_labels, "ORL", 40)

    HelpFiles.HelpPlots.plotConfusionMatrixFromEstimator(test_data, test_labels, [int(i) for i in lbls_names], knn,
                                                         "ORL", estimator_name)

    #print(classification_report(test_labels, pred, zero_division=0))
    return accuracy_score(test_labels, pred)


if __name__ == '__main__':
    nearest_neighbor_orl(5, pca=True)
    nearest_neighbor_mnist(5, pca=True)


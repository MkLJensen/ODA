from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import NearestCentroid
from MiniProject.CodeFiles.HelpFiles import ImportFiles
import MiniProject.CodeFiles.HelpFiles.HelpPlots as HelpP

estimator_name = "Nearest Class Centroid"


def nearest_centroid_mnist(pca=False):

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    ncc = NearestCentroid()
    ncc.fit(train_images, train_labels)

    pred = ncc.predict(test_images)
    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    PCA_string = "PCA" if pca else "No PCA"

    if pca:
        Hp = HelpP.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(ncc, test_images, pred, "MNIST", "NearestCentroid", 10)

    HelpP.plotConfusionMatrixFromEstimator(test_images, test_labels, [int(i) for i in lbls_names],
                                                         ncc, "MNIST", estimator_name, PCA_string)

    #print(classification_report(test_labels, pred, target_names=lbls_names))
    return accuracy_score(test_labels, pred)


def nearest_centroid_orl(pca=False):

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    ncc = NearestCentroid()
    ncc.fit(train_data, train_labels)

    pred = ncc.predict(test_data)

    lbls_names = []
    for i in range(40):
        lbls_names.append(str(i))

    PCA_string = "PCA" if pca else "No PCA"

    if pca:
        Hp = HelpP.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(ncc, test_data, pred, "ORL", "NearestCentroid", 40)

    HelpP.plotConfusionMatrixFromEstimator(test_data, test_labels, [int(i) for i in lbls_names],
                                                         ncc, "ORL", estimator_name, PCA_string)

    #print(classification_report(test_labels, pred, zero_division=0))
    return accuracy_score(test_labels, pred)

if __name__ == '__main__':
    nearest_centroid_orl(pca=True)
    nearest_centroid_mnist(pca=True)


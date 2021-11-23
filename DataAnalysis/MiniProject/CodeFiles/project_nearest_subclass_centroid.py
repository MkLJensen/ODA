from sklearn.metrics import classification_report, accuracy_score
from MiniProject.CodeFiles.HelpFiles import NearestSubclassCentroid as NSC
from MiniProject.CodeFiles.HelpFiles import ImportFiles
import HelpFiles.HelpPlots


def nearest_subclass_centroid_mnist(number_subclasses, pca=False):

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    nsc = NSC.NearestSubclassCentroid()

    nsc.fit(train_images, train_labels, number_subclasses)

    pred = nsc.predict(test_images)

    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    if pca:
        Hp = HelpFiles.HelpPlots.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(nsc, test_images, test_labels, "MNIST", 10)

    HelpFiles.HelpPlots.plotConfusionMatrixFromPreds(pred, test_labels, [int(i) for i in lbls_names], "MNIST",
                                                     "Nearest Subclass Centroid (" + str(number_subclasses) +
                                                     " Subclasses)")

    #print(classification_report(test_labels, pred, target_names=lbls_names, zero_division=0))
    return accuracy_score(test_labels, pred)


def nearest_subclass_centroid_orl(number_subclasses, pca=False):

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    nsc = NSC.NearestSubclassCentroid()

    nsc.fit(train_data, train_labels, number_subclasses)

    pred = nsc.predict(test_data)

    lbls_names = []
    for i in range(40):
        lbls_names.append(i)

    if pca:
        Hp = HelpFiles.HelpPlots.HelpPlots()
        Hp.plotScatterAndDecisionBoundaryOfClassifier(nsc, test_data, test_labels, "ORL", 40)

    HelpFiles.HelpPlots.plotConfusionMatrixFromPreds(pred, test_labels, lbls_names, "ORL",
                                                     "Nearest Subclass Centroid (" + str(number_subclasses)
                                                     + " Subclasses)")

    #print(classification_report(test_labels, pred, zero_division=0))
    return accuracy_score(test_labels, pred)


if __name__ == '__main__':
    nearest_subclass_centroid_mnist(5, pca=True)
    nearest_subclass_centroid_orl(5, pca=True)


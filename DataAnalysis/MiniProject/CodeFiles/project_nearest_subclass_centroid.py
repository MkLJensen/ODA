from sklearn.metrics import classification_report
from MiniProject.CodeFiles.HelpFiles import NearestSubclassCentroid as NSC
from MiniProject.CodeFiles.HelpFiles import ImportFiles


def nearest_subclass_centroid_mnist(pca=False):

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    nsc = NSC.NearestSubclassCentroid()

    nsc.fit(train_images, train_labels, 5)

    pred = nsc.predict(test_images)
    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    print(classification_report(test_labels, pred, target_names=lbls_names, zero_division=0))


def nearest_subclass_centroid_orl(pca=False):

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    nsc = NSC.NearestSubclassCentroid()

    nsc.fit(train_data, train_labels, 5)

    pred = nsc.predict(test_data)

    print(classification_report(test_labels, pred, zero_division=0))


if __name__ == '__main__':
    nearest_subclass_centroid_mnist(pca=False)


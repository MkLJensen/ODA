from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from MiniProject.CodeFiles.HelpFiles import ImportFiles


def nearest_neighbor_mnist(pca=False):
    n_neighbors = 5

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_images, train_labels)

    pred = knn.predict(test_images)
    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    print(classification_report(test_labels, pred, target_names=lbls_names))


def nearest_neighbor_orl(pca=False):
    n_neighbors = 40

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_data, train_labels)

    pred = knn.predict(test_data)

    print(classification_report(test_labels, pred, zero_division=0))


if __name__ == '__main__':
    nearest_neighbor_mnist(pca=False)


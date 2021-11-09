from sklearn.metrics import classification_report
from MiniProject.CodeFiles.HelpFiles import ImportFiles
from sklearn.linear_model import SGDClassifier


def perceptron_backpropagation_mnist(pca=False):

    test_images, test_labels, train_images, train_labels = ImportFiles.import_mnist(pca)

    loss = 'hinge'
    alpha = 0.05  # Margin
    learning_rate = 0.003

    sgdc = SGDClassifier(loss=loss, alpha=alpha, learning_rate='constant', eta0=learning_rate)
    sgdc.fit(train_images, train_labels)

    pred = sgdc.predict(test_images)
    lbls_names = []
    for i in range(10):
        lbls_names.append(str(i))

    print(classification_report(test_labels, pred, target_names=lbls_names))


def perceptron_backpropagation_orl(pca=False):

    train_data, test_data, train_labels, test_labels = ImportFiles.import_orl(pca)

    loss = 'hinge'
    alpha = 0.05  # Margin
    learning_rate = 0.003

    sgdc = SGDClassifier(loss=loss, alpha=alpha, learning_rate='constant', eta0=learning_rate)
    sgdc.fit(train_data, train_labels)

    pred = sgdc.predict(test_data)

    print(classification_report(test_labels, pred, zero_division=0))


if __name__ == '__main__':
    perceptron_backpropagation_orl(pca=False)


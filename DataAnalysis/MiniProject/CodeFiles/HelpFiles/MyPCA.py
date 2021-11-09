from sklearn.decomposition import PCA


def do_PCA(n_components, train_data, test_data):
    pca = PCA(n_components=2)
    pca.fit(train_data)
    train_data = pca.transform(train_data)
    test_data = pca.transform(test_data)
    return [train_data, test_data]
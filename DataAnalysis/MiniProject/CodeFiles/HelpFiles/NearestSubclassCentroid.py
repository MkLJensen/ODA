import numpy as np
from sklearn.cluster import KMeans
""" Inspiration from https://github.com/SimplisticCode/ODA-ML/blob/master/NearestSubclassCentroid.py """


class NearestSubclassCentroid:

    def __init__(self):
        self.k_means_models = []

    def fit(self, x_train, y_train, n_sub_classes):
        """ Find the number of labels https://numpy.org/doc/stable/reference/generated/numpy.unique.html """
        lbls = np.unique(y_train)

        for i in range(np.min(lbls), np.max(lbls)):
            indx = np.where(y_train == i)
            self.k_means_models.append((i, KMeans(n_sub_classes).fit(x_train[indx]).cluster_centers_))

    def predict(self, x):
        if len(self.k_means_models) <= 0:
            print("You need to call predict first")
        else:
            y_pred = np.full([len(x), ], None)
            for x_idx, x_val in enumerate(x):
                distances = []
                for idx, (i, centroids) in enumerate(self.k_means_models):
                    for sub_k in range(len(centroids)):
                        distances.append((np.linalg.norm(x_val - centroids[sub_k]), i))

                """ Sort Distances to Centroids 
                https://www.programiz.com/python-programming/methods/list/sort """
                sortedDistances = sorted(distances, key=lambda y: y[0])
                """ Take the first objects label """
                y_pred[x_idx] = int(sortedDistances[0][1])
        return y_pred.astype(int)

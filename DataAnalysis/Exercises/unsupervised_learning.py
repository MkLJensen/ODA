import numpy as np


def disToPoint(a, b):
    return np.linalg.norm(a - b)


def assignLabels(distances):
    labels = np.zeros((1, len(distances[0, :])))
    for i in range(len(distances[0, :])):
        labels[0, i] = 1 if distances[1, i] > distances[0, i] else 2
    return labels


def cluster2DimInTwoClusters(X, M):
    distances = np.zeros((2, len(X[0, :])))

    for i in range(len(X[0, :])):
        distances[0, i] = disToPoint(X[:, i], M[:, 0])
        distances[1, i] = disToPoint(X[:, i], M[:, 1])

    labels = assignLabels(distances)
    return labels, distances


def getMeanFromLabelAndX(x_cur, label):
    points1 = np.zeros((2, 1))
    points2 = np.zeros((2, 1))
    for i in range(len(label[0, :])):
        if label[:, i] == 2:
            points2 = np.append(points2, x_cur[:, i].reshape(-1, 1), axis=1)
        else:
            points1 = np.append(points1, x_cur[:, i].reshape(-1, 1), axis=1)
    points2 = points2[:, 1:]
    points1 = points1[:, 1:]
    meanPoint1 = points1.mean(axis=1).reshape(-1, 1)
    meanPoint2 = points2.mean(axis=1).reshape(-1, 1)

    return np.append(meanPoint1, meanPoint2, axis=1)


def calculateNewFuzzyCenters(X, membvals):
    M = np.zeros((2,1))
    for i in range(2):
        M_temp = np.dot(X,membvals[i,:].reshape(-1,1))
        M_temp = M_temp/sum(membvals[i,:])
        M = np.append(M, M_temp.reshape(-1, 1), axis=1)

    M = M[:, 1:]
    return M


def caculateMembershipValues(distance, my):
    dg = distance**my
    new_dist = np.zeros((2,1))
    for i in range(len(distance[0,:])):
        new_dist = np.append(new_dist, dg[:,i].reshape(-1, 1)/sum(dg[:,i]), axis=1)
    new_dist = new_dist[:, 1:]
    return new_dist


def Exercise2_2(numOfIter):
    X = np.array([[-1, 0, -0.5, -1.5, -2, 0, -1, 1, 1.3, 0.7, 2.5, 0],
                  [0, -1, -0.5, -1.5, 0, -2, -1.3, 1, 0.7, 1.3, 1, 1]])
    M = np.array([[-1, -0.9],
                  [-1, 0]])

    for i in range(numOfIter):
        labl, dist = cluster2DimInTwoClusters(X, M)
        M = getMeanFromLabelAndX(X, labl)

    print("Dist: ")
    print(dist)
    print("Labels: ")
    print(labl)
    print("M: ")
    print(M)


def Exercise2_3(numOfIter):
    X = np.array([[-1, 0, -0.5, -1.5, -2, 0, -1, 1, 1.3, 0.7, 2.5, 0],
                  [0, -1, -0.5, -1.5, 0, -2, -1.3, 1, 0.7, 1.3, 1, 1]])
    M = np.array([[-1, -0.9],
                  [-1, 0]])
    for i in range(numOfIter):
        labl, dist = cluster2DimInTwoClusters(X, M)
        membval = caculateMembershipValues(dist, -2)
        M = calculateNewFuzzyCenters(X,membval)

    print("Dist: ")
    print(dist)
    print("Memb: ")
    print(membval)
    print("M: ")
    print(M)


def Exercise2_4():
    X = np.array([[-1, 0, -0.5, -1.5, -2, 0, -1, 1, 1.3, 0.7, 2.5, 0],
                  [0, -1, -0.5, -1.5, 0, -2, -1.3, 1, 0.7, 1.3, 1, 1]])

    m = X.mean(axis=1)
    X_centered = X-np.ones((2,12))*m.reshape(-1,1)

    ss = X.std(axis=1,ddof=1)
    X_standard = X_centered/(np.ones((2,12))*ss.reshape(-1,1))

    X_norm = np.zeros((2,1))
    for i in range(len(X[0,:])):
        X_norm = np.append(X_norm, (X[:,i] / np.linalg.norm(X[:,i])).reshape(-1,1),axis=1)
    X_norm = X_norm[:, 1:]

    print("s: ")
    print(ss)
    print("m: ")
    print(m)
    print("Centered X: ")
    print(X_centered)
    print("Standard X: ")
    print(X_standard)
    print("Normalized X: ")
    print(X_norm)


def Exercise2_5():
    X = np.array([[1, 2, 2, 3, 3, 4],
                  [1, 2, 3, 2, 3, 4]])
    m = X.mean(axis=1)
    X_centered = X - np.ones((2, len(X[0,:]))) * m.reshape(-1, 1)

    cov_matrix = np.dot(X_centered, np.transpose(X_centered))

    w,v = np.linalg.eig(cov_matrix)
    # since 10 is max eigenvalue, take its eigenvector
    x_proj = np.dot(np.transpose(v[:,0]),X)

    print("Projected X")
    print(x_proj)

if __name__ == '__main__':
    Exercise2_5()

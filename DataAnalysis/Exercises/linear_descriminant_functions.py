import numpy as np


def exercise4_1():
    # Show that maximizing the LDA criterion in Eq(4.22) is equivalent to minimizing J2(w)
    # We use S_t = S_w + S_b
    # J(W) + 1 = (Tr(W^T*S_b*W))/(Tr(W^T*S_w*W)) + (Tr(W^T*S_w*W))/(Tr(W^T*S_w*W)) = 1/(J2(w))
    # For better representation see AIS(21)
    print()


# Super Naive Way, where could return more value if it finds more
def find_lbls_call(arr_find, what_to_find, x_len):
    dis_lbl = np.zeros((1, x_len))
    for i in range(len(what_to_find[0, :])):
        dis_lbl[0, i] = np.where(what_to_find[:, i] == arr_find[i])[0]
    return dis_lbl


def nearest_centroid_class(c1, c2, x, k):
    M = np.zeros((2, 2))
    M[:, 0] = np.mean(c1, axis=1)
    M[:, 1] = np.mean(c2, axis=1)

    Dis = np.zeros((2, len(x[0, :])))

    for i in range(k):
        for j in range(len(x[0, :])):
            Dis[i, j] = np.linalg.norm(x[:, j] - M[:, i])

    dis_min = Dis.min(axis=0)

    dis_lbl = find_lbls_call(dis_min, Dis, len(x[0, :]))

    return [dis_min, dis_lbl]


def nearest_neighbors_class(c1, c2, x, k):
    x_len = len(x[0, :])
    c1_len = len(c1[0, :])
    Dnn = np.zeros((2, x_len))
    dis = np.zeros((1, c1_len))
    for i in range(x_len):
        for j in range(c1_len):
            dis[0, j] = np.linalg.norm(x[:, i] - c1[:, j])

        Dnn[0, i] = dis.min()

        for l in range(len(c2[0, :])):
            dis[0, l] = np.linalg.norm(x[:, i] - c2[:, l])

        Dnn[1, i] = dis.min()
    dis_min = Dnn.min(axis=0)

    dis_lbl = find_lbls_call(dis_min, Dnn, x_len)

    return [dis_min, dis_lbl]


def exercise4_2(c1, c2, X):

    K = 2

    dis_min_nc, dis_lbl_nc = nearest_centroid_class(c1, c2, X, K)
    print("Nearest Centroid Classifier: " + np.array_str(dis_min_nc))
    print("Nearest Centroid Classifier Labels: " + np.array_str(dis_lbl_nc))

    dis_min_nn, dis_lbl_nn = nearest_neighbors_class(c1, c2, X, K)
    print("Nearest Neighbor Classifier: " + np.array_str(dis_min_nn))
    print("Nearest Neighbor Classifier Labels: " + np.array_str(dis_lbl_nn))


# calculate the 1d feature space determined by applying the Fisher Discriminant Analysis
def exercise4_3(c1, c2):
    total_mean = np.mean(np.append(c1, c2, axis=1), axis=1)
    m1 = np.mean(c1, axis=1)
    m2 = np.mean(c2, axis=1)

    cl_center1m = c1 - m1.reshape(-1, 1) * np.ones((1, len(c1[0, :])))
    cl_center2m = c2 - m2.reshape(-1, 1) * np.ones((1, len(c2[0, :])))

    Sw = np.dot(cl_center1m, np.transpose(cl_center1m)) + np.dot(cl_center2m, np.transpose(cl_center2m))
    Sb = np.outer((m1-m2), np.transpose((m1-m2)))

    Sw_inv = np.linalg.inv(Sw)

    w, v = np.linalg.eig(np.dot(Sw_inv, Sb))

    # Select Max Eigenvalue
    maxeig = v[:, 1]

    X = np.append(c1, c2, axis=1)
    x = np.dot(maxeig.reshape(1, -1), X)

    print("Projected Data: " + np.array_str(x))


def exercise4_4(c1, c2, x):

    w0 = np.transpose(np.array([0.1, 0.1, 0]))
    ln = 0.01
    maxIter = 10

    X = np.append(np.append(c1, c2, axis=1), np.ones((1, len(c1[0, :])+len(c2[0, :]))), axis=0)
    lbls = np.transpose(np.append(np.ones((len(c1[0, :]), 1)),
                        np.full((len(c2[0, :]), 1), -1), axis=0))

    N = len(X[0, :])
    O = np.zeros((maxIter, N))

    w = w0

    DeltaAll = []
    wAll = []

    for i in range(maxIter):
        XX = np.full((3, 1), -255)
        LL = np.full((1, 1), -255)
        for j in range(N):
            o = np.dot(np.transpose(w), X[:, j])
            O[i, j] = o
            pos_o = lbls[0, j] * o
            if pos_o < 0:
                XX = np.append(XX, X[:, j].reshape(-1, 1), axis=1)
                LL = np.append(LL, lbls[:, j].reshape(-1, 1), axis=1)
                print(j+1)

        # Remove start entries
        XX = XX[:, 1:]
        LL = LL[:, 1:]

        #Update w
        Delta = np.zeros((3, 1))
        for j in range(len(XX[0, :])):
            Delta = Delta + LL[:, j].reshape(-1, 1)*XX[:, j].reshape(-1, 1)

        w = w.reshape(-1, 1)+ln*Delta
        DeltaAll.append(Delta)
        wAll.append(w)

        if len(XX[0, :]) == 0:
            break

    # TEST
    X = np.append(x, np.ones((1, len(x[0, :]))), axis=0)
    N = len(X[0, :])
    l = np.zeros((1, N))
    o = np.zeros((N, 1))
    for i in range(N):
        o[i, 0] = np.dot(np.transpose(w), X[:, i].reshape(-1, 1))
        if o[i, 0] > 0:
            l[0, i] = 1
        else:
            l[0, i] = -1

    print("Values of Network is " + np.array_str(np.transpose(o)))
    print("Labels are: " + np.array_str(l))


if __name__ == '__main__':
    c1 = np.array([[-1, 0, -0.5, -1.5, -2, 0, -1],
                   [0, -1, -0.5, -1.5, 0, -2, -1.3]])

    c2 = np.array([[1, 1.3, 0.7, 2.5, 0],
                   [1, 0.7, 1.3, 1, 1]])

    # Classify the following samples using the Nearest Centroid and Nearest Neighbor classifiers
    X = np.array([[0, 1, -1, 0.7, -0.2],
                  [0, 1, 0, -0.2, 1.5]])

    exercise4_4(c1, c2, X)

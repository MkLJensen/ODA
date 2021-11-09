import numpy as np

c1 = np.array([[-1, 0, -0.5, -1.5, -2, 0, -1],
                [0, -1, -0.5, -1.5, 0, -2, -1.3]])

c2 = np.array([[1, 1.3, 0.7, 2.5, 0],
                [1, 0.7, 1.3, 1, 1]])

X = np.array([[0, 1, -1, 0.7, -0.2],
                [0, 1, 0, -0.2, 1.5]])

lambdarray = np.array([[0, 1],
                       [1, 0]])

# Calculate Priori Probabilities of each class
N_c1 = np.size(c1, axis=1)
N_c2 = np.size(c2, axis=1)

P_c1 = N_c1/(N_c1+N_c2)
P_c2 = N_c2/(N_c1+N_c2)

# Mean Probability
m1 = np.mean(c1, axis=1)
m2 = np.mean(c2, axis=1)

N = np.size(X, axis=1)

p_x_c1 = np.zeros(N)
p_x_c2 = np.zeros(N)
P_c1_x = np.zeros(N)
P_c2_x = np.zeros(N)
Diff = np.zeros(N)
R_a1_x = np.zeros(N)
R_a2_x = np.zeros(N)

def Exercise3_1():
    for i in range(N):
        # Distance from samples to mean
        d_x_c1 = np.linalg.norm(X[:, i]-m1)
        d_x_c2 = np.linalg.norm(X[:, i]-m2)

        # If p(x | c_k) is a function of the relative distance
        # p_x_c1[i] = d_x_c1 / (d_x_c1 + d_x_c2)
        # p_x_c2[i] = d_x_c2 / (d_x_c1 + d_x_c2)

        # If p(x | c_k) is a function of the relative inverse distance
        # p_x_c1(i) = (1 / d_x_c1) / ((1 / d_x_c1) + (1 / d_x_c2));
        # p_x_c2(i) = (1 / d_x_c2) / ((1 / d_x_c1) + (1 / d_x_c2));

        # If p(x | c_k) is a function of the relative exponential distance
        p_x_c1[i] = np.exp(-d_x_c1) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))
        p_x_c2[i] = np.exp(-d_x_c2) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))

        # calculate the probability of each class , given x

        #
        P_c1_x[i] = p_x_c1[i] * P_c1
        P_c2_x[i] = p_x_c2[i] * P_c2

        # The Diff will tell us what class each X belongs to following whether it has a - or +
        Diff[i] = P_c1_x[i] - P_c2_x[i]  # if Diff > 0 --> c_1, else c_2

    print(Diff)


def Exercise3_2():
    for i in range(N):
        # Distance from samples to mean
        d_x_c1 = np.linalg.norm(X[:, i] - m1)
        d_x_c2 = np.linalg.norm(X[:, i] - m2)

        # If p(x | c_k) is a function of the relative exponential distance
        p_x_c1[i] = np.exp(-d_x_c1) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))
        p_x_c2[i] = np.exp(-d_x_c2) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))

        # calculate the probability of each class , given x
        P_c1_x[i] = p_x_c1[i] * P_c1
        P_c2_x[i] = p_x_c2[i] * P_c2

        R_a1_x[i] = P_c1_x[i] * lambdarray[0, 0] + P_c2_x[i] * lambdarray[0, 1]
        R_a2_x[i] = P_c1_x[i] * lambdarray[1, 0] + P_c2_x[i] * lambdarray[1, 1]

        # The Diff will tell us what class each X belongs to following whether it has a - or +
        Diff[i] = R_a2_x[i] - R_a1_x[i]  # if Diff > 0 --> c_1, else c_2

    print(Diff)


def Exercise3_3():

    risks = np.array([[0.4, 0.8],
                      [0.6, 0.2]])

    for i in range(N):
        # Distance from samples to mean
        d_x_c1 = np.linalg.norm(X[:, i] - m1)
        d_x_c2 = np.linalg.norm(X[:, i] - m2)

        # If p(x | c_k) is a function of the relative exponential distance
        p_x_c1[i] = np.exp(-d_x_c1) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))
        p_x_c2[i] = np.exp(-d_x_c2) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))

        # calculate the probability of each class , given x
        P_c1_x[i] = p_x_c1[i] * P_c1
        P_c2_x[i] = p_x_c2[i] * P_c2

        R_a1_x[i] = P_c1_x[i] * risks[0, 0] + P_c2_x[i] * risks[0, 1]
        R_a2_x[i] = P_c1_x[i] * risks[1, 0] + P_c2_x[i] * risks[1, 1]

        # The Diff will tell us what class each X belongs to following whether it has a - or +
        Diff[i] = R_a2_x[i] - R_a1_x[i]  # if Diff > 0 --> c_1, else c_2

    print(Diff)


def Exercise3_4():
    covmatrix = np.array([[1, 0],
                          [0, 2]])
    invS = np.linalg.pinv(covmatrix)

    for i in range(N):

        d_x_c1 = np.dot(np.dot(np.transpose((X[:, i] - m1)), invS), (X[:, i] - m1))
        d_x_c2 = np.dot(np.dot(np.transpose((X[:, i] - m2)), invS), (X[:, i] - m2))
        # If p(x | c_k) is a function of the relative exponential distance
        p_x_c1[i] = np.exp(-d_x_c1) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))
        p_x_c2[i] = np.exp(-d_x_c2) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))

        # calculate the probability of each class , given x
        P_c1_x[i] = p_x_c1[i] * P_c1
        P_c2_x[i] = p_x_c2[i] * P_c2

        # The Diff will tell us what class each X belongs to following whether it has a - or +
        Diff[i] = P_c1_x[i] - P_c2_x[i]  # if Diff > 0 --> c_1, else c_2

    print(Diff)


def Exercise3_5():

    Xm1 = c1 - (m1.reshape(-1, 1) * np.ones((1, N_c1)))
    Xm2 = c2 - (m2.reshape(-1, 1) * np.ones((1, N_c2)))

    S1 = np.dot(Xm1, np.transpose(Xm1))
    S2 = np.dot(Xm2, np.transpose(Xm2))

    # From Exercise 1/2 * (Sum1+Sum2)
    S = (S1+S2)/2
    invS = np.linalg.pinv(S)

    for i in range(N):
        d_x_c1 = np.dot(np.dot(np.transpose((X[:, i] - m1)), invS), (X[:, i] - m1))
        d_x_c2 = np.dot(np.dot(np.transpose((X[:, i] - m2)), invS), (X[:, i] - m2))
        # If p(x | c_k) is a function of the relative exponential distance
        p_x_c1[i] = np.exp(-d_x_c1) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))
        p_x_c2[i] = np.exp(-d_x_c2) / (np.exp(-d_x_c1) + np.exp(-d_x_c2))

        # calculate the probability of each class , given x
        P_c1_x[i] = p_x_c1[i] * P_c1
        P_c2_x[i] = p_x_c2[i] * P_c2

        # The Diff will tell us what class each X belongs to following whether it has a - or +
        Diff[i] = P_c1_x[i] - P_c2_x[i]  # if Diff > 0 --> c_1, else c_2

    print(Diff)


if __name__ == '__main__':
    Exercise3_5()

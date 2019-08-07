import numpy as np
import scipy


def ssvd(z, k):
    zt = np.transpose(z)
    s = []
    v = []

    # calculate zt*z sequentially
    for i in range(0, len(z[0])):
        v.append(0)
    for i in range(0, len(z[0])):
        s.append(v)
    for i in range(0, len(z[0])):
        for j in range(0, len(z[0])):
            temp = 0
            for k in range(0, len(z)):
                temp = temp + zt[i][k] * z[k][j]
            s[i][j] = temp
    # print (s)

    # eigendecomposition
    # val vec
    B, sigma = np.linalg.eig(s)
    sigma = scipy.linalg.fractional_matrix_power(sigma, 0.5)
    sigma_inverse = scipy.linalg.fractional_matrix_power(sigma, -1)
    R = np.matmul(sigma_inverse, B)
    A = []
    for q in z:
        A.append(np.matmul(q, R))
    print("---------------------B is compute : " + str(B))
    print("---------------------sigma is compute : " + str(sigma))
    print("---------------------A is compute : " + str(A))
    print("done")
    return A, B, sigma

import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import svd


def show_result(z,A, sigma, B):
    # print("---------------------z is compute : " + str(z))
    # print("---------------------B is compute : " + str(B))
    # print("---------------------sigma is compute : " + str(sigma))
    # print("---------------------A is compute : " + str(A))
    print("SSVD done")
    return svd(z)


def ssvd(z):

    zt = np.transpose(z)
    s = []
    v = []
    s = np.matmul(zt,z)
    B, sigma = np.linalg.eig(s)
    sigma = fractional_matrix_power(sigma, 0.5)
    sigma_inverse = fractional_matrix_power(sigma, -1)
    R = np.matmul(sigma_inverse, B)
    A = []
    for zi in z:
        A.append(np.matmul(zi, R))
    return show_result(z,A,sigma,B)

# z = [[1,2,3],[2,3,4],[4,5,6],[5,6,7]]
# k = 2
# a,sigma,b =ssvd(z,k)
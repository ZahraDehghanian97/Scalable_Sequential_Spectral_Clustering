import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import svd


def ssvd(z):
    print("SSVD start")
    zt = np.transpose(z)
    s = np.matmul(zt,z)
    print(s.shape)
    sigma,B = np.linalg.eig(s)
    print(B.shape)
    print(sigma.shape)
    sigma = np.diag(sigma)
    print(sigma.shape)
    sigma = fractional_matrix_power(sigma, 0.5)
    sigma_inverse = fractional_matrix_power(sigma, -1)
    print(sigma_inverse.shape)
    R = np.matmul(B,sigma_inverse)
    A = np.matmul(z,R)
    # return show_result(z, A, sigma, B)
    return A , sigma ,B

def show_result(z, A, sigma, B):
    # print("---------------------z is compute : " + str(z))
    # print("---------------------B is compute : " + str(B))
    # print("---------------------sigma is compute : " + str(sigma))
    # print("---------------------A is compute : " + str(A))
    print("SSVD done")
    return svd(z)

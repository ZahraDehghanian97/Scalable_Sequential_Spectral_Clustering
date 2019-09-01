import numpy as np
from scipy.linalg import fractional_matrix_power
from scipy.linalg import svd

def ssvd(z, k):
    A,sigma,B = svd(z)
    # zt = np.transpose(z)
    # s = []
    # v = []
    # # for i in range(0, len(z[0])):
    # #     v.append(0)
    # # for i in range(0, len(z[0])):
    # #     s.append(v)
    # # for i in range(0, len(z[0])):
    # #     for j in range(0, len(z[0])):
    # #         temp = 0
    # #         for k in range(0, len(z)):
    # #             temp = temp + zt[i][k] * z[k][j]
    # #         s[i][j] = temp
    # s = np.matmul(zt,z)
    # B, sigma = np.linalg.eig(s)
    # # sigma = sigma[:, np.argsort(B)]
    # # B = B[np.argsort(B)]
    # # sigma = sigma [:k]
    # # B = B[:k]
    # sigma = fractional_matrix_power(sigma, 0.5)
    # sigma_inverse = fractional_matrix_power(sigma, -1)
    # R = np.matmul(sigma_inverse, B)
    # A = []
    # for q in z:
    #     A.append(np.matmul(q, R))
    # print("---------------------B is compute : " + str(B))
    # print("---------------------sigma is compute : " + str(sigma))
    # print("---------------------A is compute : " + str(A))
    # print("done")
    return A, B, sigma

# z = [[1,2,3],[2,3,4],[4,5,6],[5,6,7]]
# k = 2
# a,sigma,b =ssvd(z,k)
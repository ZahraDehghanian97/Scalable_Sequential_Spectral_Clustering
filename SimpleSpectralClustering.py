import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

random_state = 21
X_mn, y_mn = make_moons(150, noise=.07)
dot_size = 25

from sklearn.neighbors import radius_neighbors_graph
A = radius_neighbors_graph(X_mn, 0.4, mode='distance', metric='minkowski', p=2, metric_params=None, include_self=False)
A = A.toarray()
# print(A)

from scipy.sparse import csgraph
L = csgraph.laplacian(A, normed=False)
eigval, eigvec = np.linalg.eig(L)
# z = np.partition(eigval, 1)
# print(np.where(eigval == z[1]))  # the second smallest eigenvalue

y_spec = eigvec[:, 1].copy()
y_spec[y_spec < 0] = 0
y_spec[y_spec > 0] = 1

fig, ax = plt.subplots(figsize=(6,4))
ax.scatter(X_mn[:, 0], X_mn[:, 1],c=y_spec ,s=dot_size)
plt.show()

# ready to use varsion
from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=2, affinity='nearest_neighbors',
                          assign_labels='kmeans')
labelsS = model.fit_predict(X_mn)
fig, ax = plt.subplots(figsize=(6,4))
# ax.set_title('kernal transform to higher dimension\nlinear separation is possible', fontsize=18, fontweight='demi')
plt.scatter(X_mn[:, 0], X_mn[:, 1], c=labelsS, s=dot_size)
plt.show()
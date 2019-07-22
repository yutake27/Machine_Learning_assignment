import numpy as np
import matplotlib.pyplot as plt

# condition
A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])
eta = 1/np.max(np.linalg.eig(2*A)[0])
lam = 2
w = np.array([[ 3],
              [-1]])

def PG(w):
    w_t = np.zeros(w.shape)
    for i in range(len(w)):
        if w[i] - eta*(np.dot(A+A.T, w-mu)[i]+lam) >0:
            w_t[i] = w[i] - eta*(np.dot(A+A.T, w-mu)[i]+lam)
        elif w[i] - eta*(np.dot(A+A.T, w-mu)[i]) == 0:
            w_t[i] = 0
        else:
            w_t[i] = w[i] - eta*(np.dot(A+A.T, w-mu)[i] - lam)
    return w_t


w_hat_lam = []
lams = np.arange(0, 6, 0.1)

for lam in lams:
    ### implement PG ###
    w = np.array([[ 3],
              [-1]])
    num_iter = 100
    for t in range(num_iter):
        w = PG(w)
    w_hat_lam.append(w.T)
    ###
w_hat_lam = np.vstack(w_hat_lam)

plt.plot(lams, w_hat_lam[:,0], label='w0')
plt.plot(lams, w_hat_lam[:,1], label='w1')
plt.legend()
plt.xlabel('lambda')
plt.ylabel('w')
plt.show()
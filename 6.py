import numpy as np
import matplotlib.pyplot as plt

#dataset3
np.random.seed(1234)
m = 20
n = 40
r = 2
A = np.ravel(np.dot(np.random.rand(m, r), np.random.rand(r, n)))
ninc = 100
Q = np.random.permutation(m * n)[:ninc]
A[Q] = None
A = A.reshape(m, n)

# plt.pcolor(A)
# plt.show()

def projection(Z, A):
    Z = np.copy(Z)
    Z[np.isnan(A)] = 0
    return Z

def prox(Z, eta=0.01, lam=0.01):
    u, s, vh = np.linalg.svd(Z)
    s = np.max((np.zeros(len(s)), s-eta*lam), axis = 0)
    sigma = np.zeros((len(u),len(vh)))
    for i in range(len(s)):
        sigma[i][i] = s[i]
    return u.dot(sigma).dot(vh)

def Loss(Z, A, lam=0.01):
    Z, A = projection(Z,A), projection(A,A)
    psi = np.sum((Z-A)**2)
    _, s, _ = np.linalg.svd(Z)
    omega = np.sum(s)
    return psi/2+lam*omega

u, s, vh = np.linalg.svd(projection(A,A))
print(u[0])
Z = np.random.rand(m,n)
u, s, vh = np.linalg.svd(Z)
print(u[0])

Z_history = []
loss_history = []
num_iter = 500
eta = 0.01
for t in range(num_iter):
    Z_history.append(Z)
    loss_history.append(Loss(Z, A))
    Z = prox(Z-eta*(projection(Z,A)-projection(A,A)))
    if t>2:
        if loss_history[-2]-loss_history[-1]<0.0001:
            print(t)
            break

# plt.plot(loss_history)
# plt.xlabel('t')
# plt.ylabel('Loss')
# print(A)
# print(Z-projection(A,A))
plt.pcolor(Z)
plt.show()
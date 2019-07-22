# requirements
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv

# dataset 4
np.random.seed(123)
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 -1

def mul(x, y):
    return np.array([y[i]*x[i] for i in range(len(y))])

def hinge_reg(a, x, y):
    w = 1/(2*lam)*np.sum(mul(a*y, x), axis=0)
    ywx = y*np.sum(w*x, axis=1)
    return np.sum(np.maximum(np.zeros(len(y)), 1-ywx)) + lam*np.dot(w, w)

def negLagFunc(a, x, y):
    w = 1/(2*lam)*np.sum(mul(a*y, x), axis=0)
    return 1/(4*lam)*np.dot(w.T, w) - np.sum(a)

def projection(a):
    a_hat = np.array([1 if a[i]>1 else a[i] if a[i]>=0 else 0 for i in range(len(a))])
    return a_hat

def a_t(a, K, eta):
    return projection(a-eta*(1/(2*lam)*np.dot(K, a)-1))

LagFunc_hist = []
hinge_reg_hist = []
lam = 0.01
eta = 0.0001
a = np.random.rand(len(x_d4))
K = np.array([[y_d4[i]*y_d4[j]*np.dot(x_d4[i], x_d4[j]) for i in range(len(x_d4))] for j in range(len(x_d4))])

num_iter = 400
for t in range(num_iter):
    LagFunc_hist.append(negLagFunc(a, x_d4, y_d4))
    hinge_reg_hist.append(hinge_reg(a, x_d4, y_d4))
    a = a_t(a, K, eta)



# validity
num_iter = 50
plt.plot(hinge_reg_hist[:num_iter], 'r')
plt.xlabel('iteration')
plt.ylabel('sum of hinge loss function and regularization')
plt.show()

plt.plot(LagFunc_hist[:num_iter])
plt.ylabel('negative dual Lagrange function')
plt.show()


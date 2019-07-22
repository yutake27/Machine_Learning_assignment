import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt

np.random.seed(1)
# dataset 4
n = 200
x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
# print(x_d4)
y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
y_d4 = 2 * y_d4 - 1
x_d4 = np.insert(x_d4, x_d4.shape[1], np.ones(x_d4.shape[0]), axis=1)


def mul(x, y):
    """行列の型が一致しないもののかけ算を行う

    Arguments:
        x {np.array} -- 次元が少ない方 ex shape (200,)
        y {np.array} -- 次元が多い方 ex shape(200,4)
    """
    return np.array([y[i]*x[i] for i in range(len(y))])


# implement batch steepsest gradient method here
def J(w):
    return np.mean(np.log(1+np.exp(-y_d4*np.sum(w*x_d4, axis=1)))) + lam*np.sum(w**2)

def NablaJ(w):
    p = mul(((np.exp(-y_d4*np.sum(w*x_d4, axis=1)))/(1+np.exp(-y_d4*np.sum(w*x_d4, axis=1)))), -1*mul(y_d4, x_d4))
    return np.mean(p, axis=0)+2*lam*w

lam = 0.01
w = np.random.rand(x_d4.shape[1])
w2 = np.copy(w)
w_hist_batch = []
loss_hist_batch = [] # to store the history of loss J(w_t)

num_iter = 300
for t in range(1, num_iter):
    d = -NablaJ(w)
    w_hist_batch.append(w)
    loss_hist_batch.append(J(w))
    w += d

print(w)
print(loss_hist_batch[-1])




# implement Newton based method here
def hessianJ(w):
    p = (np.exp(-y_d4*np.sum(w*x_d4, axis=1)))/((1+np.exp(-y_d4*np.sum(w*x_d4, axis=1)))**2)
    x_xt = np.array([np.outer(x_d4_i, x_d4_i) for x_d4_i in x_d4])
    hes = np.mean(np.array([p_i*x_xt_i for p_i, x_xt_i in zip(p, x_xt)]), axis=0) + 2*lam*np.eye(len(w))
    return hes


# w = init*np.ones(x_d4.shape[1])
w = w2
w_hist_newton = []
loss_hist_newton = []

lam = 0.01
for t in range(1, num_iter):
    Nabla = NablaJ(w)
    hes = hessianJ(w)
    d = np.linalg.solve(hes, -Nabla)
    w_hist_newton.append(w)
    loss_hist_newton.append(J(w))
    w += d*1.0/np.sqrt(t+10)
print(w)
print(loss_hist_newton[-1])


# compare diff
num_iter = 50
plt.plot(np.abs(loss_hist_batch[:num_iter]-loss_hist_batch[-1]), 'bo-', linewidth=0.5, markersize=1, label='seepest')
plt.plot(np.abs(loss_hist_newton[:num_iter]-loss_hist_batch[-1]), 'ro-', linewidth=0.5, markersize=1, label='newton')
plt.legend()
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('diff from J(w_hat)')
plt.show()
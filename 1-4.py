import numpy as np
import matplotlib.pyplot as plt

# dataset 5
n = 200
np.random.seed(100)
x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
W = np.array([[ 2,  -1, 0.5,],
              [-3,   2,   1,],
              [ 1,   2,   3]])
y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                        + 0.5 * np.random.randn(n, 3), axis=1)
x_d5 = np.c_[x_d5, np.ones(len(x_d5))]

def softmax(x):
    ex = np.exp(x-np.max(x))  #prevent overflow
    return ex/np.sum(ex)

def get_onehot(y, class_num):
    return np.eye(class_num)[y]

def loss(y, y_pred):
    #prevent overflow
    return -np.sum(y*np.log(np.maximum(y_pred, 1.0e-20*np.ones(y.shape))))


# implement steepest gradient method
y_d5 = get_onehot(y_d5, 3)
w = np.random.randn(3, x_d5.shape[1])
w2 = np.copy(w)

lam=0.01
eta = 0.05
loss_hist_steepest = []
num_iter = 300
for t in range(num_iter):
    y_pred_array = []
    nabla_sum = np.zeros(w.shape)
    for x, y in zip(x_d5, y_d5):
        y_pred = softmax(np.dot(w, x))
        y_pred_array.append(y_pred)
        nabla_sum += np.outer(y-y_pred, x)
    w += eta*(nabla_sum-2*lam*w)
    loss_hist_steepest.append(loss(y_d5, y_pred_array))

print(w)
print(loss_hist_steepest[-1])


# implement newton based method
w = w2

loss_hist_newton = []
eta = 3
for t in range(num_iter):
    y_pred_array = []
    hessian_sum = np.zeros((len(w), x_d5.shape[1], x_d5.shape[1]))
    nabla_sum = np.zeros((len(w), x_d5.shape[1]))
    for x, y in zip(x_d5, y_d5):
        y_pred = softmax(np.dot(w, x))
        y_pred_array.append(y_pred)
        hessian_sum += [(1-y_pred[i])*np.dot(x.reshape(len(x),-1), x.reshape(-1,len(x))) for i in range(len(y))]
        nabla_sum += np.outer(y-y_pred, x)
    d = np.array([np.linalg.solve(hessian_sum[i]+2*lam*np.eye(w.shape[1]), nabla_sum[i]-2*lam*w[i]) for i in range(len(w))])
    w += eta*d
    loss_hist_newton.append(loss(y_d5, y_pred_array))

print(w)
print(loss_hist_newton[-1])

# compare diff
num_iter = 100
plt.plot(np.abs(loss_hist_steepest[:num_iter]-loss_hist_steepest[-1]), 'bo-', linewidth=0.5, markersize=1, label='seepest')
plt.plot(np.abs(loss_hist_newton[:num_iter]-loss_hist_newton[-1]), 'ro-', linewidth=0.5, markersize=1, label='newton')
plt.legend()
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('diff from J(w_hat)')
plt.show()
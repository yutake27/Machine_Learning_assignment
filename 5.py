import numpy as np
import matplotlib.pyplot as plt

# dataset 1
np.random.seed(123)
n = 200
x_d1 = 3 * (np.random.rand(n, 2)-0.5)
radius = x_d1[:,0]**2 + x_d1[:,1]**2
y_d1 = (radius > 0.7 + 0.1 * np.random.randn(n)) &( radius < 2.2 + 0.1 * np.random.randn(n))
y_d1 = 2 * y_d1 -1


def kernel(x1, x2, alpha):
    return np.exp(-1*alpha*np.dot(x1-x2, x1-x2))

def mul(x, y):
    return np.array([y[i]*x[i] for i in range(len(y))])

def hinge_reg(a, x, y, lam):
    w = 1/(2*lam)*np.sum(mul(a*y, x), axis=0)
    ywx = y*np.sum(w*x, axis=1)
    return np.sum(np.maximum(np.zeros(len(y)), 1-ywx)) + lam*np.dot(w, w)

def negLagFunc(a, x, y, lam):
    w = 1/(2*lam)*np.sum(mul(a*y, x), axis=0)
    return 1/(4*lam)*np.dot(w.T, w) - np.sum(a)

def projection(a):
    a_hat = np.array([1 if a[i]>1 else a[i] if a[i]>=0 else 0 for i in range(len(a))])
    return a_hat

def a_t(a, K, lam, eta):
    return projection(a-eta*(1/(2*lam)*np.dot(K, a)-1))

def get_a(x_data, y_data, alpha):
    LagFunc_history = []
    hinge_reg_history = []
    a_history = []
    lam = 0.5
    eta = 0.01
    a = np.random.rand(len(x_d1))
    num_iter = 500
    K = np.array([[y_data[i]*y_data[j]*kernel(x_data[i], x_data[j], alpha=alpha) for i in range(len(x_data))] for j in range(len(x_d1))])
    for t in range(num_iter):
        LagFunc_history.append(negLagFunc(a, x_data, y_data, lam))
        hinge_reg_history.append(hinge_reg(a, x_data, y_data, lam))
        a_history.append(a)
        a = a_t(a, K, lam, eta)
        if t>10:
            if LagFunc_history[-1]-LagFunc_history[-2]>0:
                a = a_history[-2]
                break
    return a

def predict(x, a, x_data, y_data, alpha):
    sum = np.sum([a[i]*y_data[i]*kernel(x_data[i], x, alpha) for i in range(len(x_data))])
    return 1 if sum > 0 else -1

def get_accuracy(a, x_data, y_data):
    y_pre = [predict(x, a, x_data, y_data, alpha) for x in x_data]
    accuracy = np.mean([1 if y_pre[i]==y_data[i]  else 0 for i in range(len(y_data))])
    return accuracy

def plot_decision(a, x_data, y_data, alpha, resolution = 0.05):
    plt.plot(x_data[y_data < 0, 0],  x_data[y_data < 0, 1], 'bs')
    plt.plot(x_data[y_data > 0, 0],  x_data[y_data > 0, 1], 'ro')
    x1_min, x1_max = x_data[:,0].min()-1, x_data[:,0].max()+1
    x2_min, x2_max = x_data[:,1].min()-1, x_data[:,1].max()+1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                            np.arange(x2_min, x2_max, resolution))

    x_pre = np.array([xx1.ravel(), xx2.ravel()]).T
    y_pre = np.array([predict(x, a, x_data, y_data, alpha) for x in x_pre])
    y_pre = y_pre.reshape(xx1.shape)
    accuracy = get_accuracy(a, x_data, y_data)
    plt.contour(xx1, xx2, y_pre)
    plt.title('alpha: {}, accuracy: {}'.format(alpha, accuracy), fontsize=18)
    plt.show()


alpha = [0.1, 1, 10, 20]
for alpha in alpha:
    a = get_a(x_d1, y_d1, alpha)
    plot_decision(a, x_d1, y_d1, alpha)
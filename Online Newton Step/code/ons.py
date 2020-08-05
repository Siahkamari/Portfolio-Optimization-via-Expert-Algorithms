import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op

def project(y, A):
    n = y.size

    theta = np.linalg.solve(A, np.ones(n))


    c = np.zeros(2*n + 2)
    A1 = np.concatenate((np.ones([1,n]),np.zeros([1,n + 2])), axis = 1)
    A2 = np.concatenate((np.eye(n),-np.eye(n), theta.reshape(-1,1), -theta.reshape(-1,1)), axis = 1)
    A_eq = np.concatenate((A1, A2), axis = 0)
    b_eq = np.concatenate(([1],y))

    res = op.linprog(c = c , A_eq = A_eq, b_eq = b_eq, method='revised simplex')

    return res.x[0:n]

# Data
n = 10
T = 2000

r = 1 + 0.01*np.random.randn(n,T) + 0.001*np.random.randn(n,1) 
stocks =  np.cumprod(r, axis = 1) 

# Parameters
gamma = 2
epsilon = 1/8

# Initialization
A = epsilon*np.eye(n)
x = np.ones(n)/n
f = np.zeros(T)

# Algorithms
# for t in range(T-1):
#     r = stocks[:,t+1]/stocks[:,t]
#     f[t] = - np.log(np.dot(x,r))
#     grad = - r/np.dot(x, r)

#     # ONS
#     A = A + np.outer(grad, grad)
#     y = x - 1/gamma * np.linalg.solve(A, grad)
#     x = project(y, A)

#     # GD
#     x = x - 1/gamma * grad
#     x = project(x, np.eye(n))

## parameter free expert
sum_g_tild = np.zeros(n)
sum_g_w = np.zeros(n)
x_0 = np.ones(n)/n
for t in range(T-1):

    w = sum_g_tild /(t+T/2)*(1+sum_g_w)
    x = x_0 * np.maximum(w, 0)

    if np.sum(x) > 0:
        x = x / np.sum(x)
    else:
        x = x_0

    r = stocks[:,t+1]/stocks[:,t]
    f[t] = - np.log(np.dot(x,r))
    grad = r/np.dot(x, r)

    g_tild = grad - np.dot(grad, x)
    g_tild[w<0] = np.maximum(g_tild[w<0], 0)

    sum_g_tild += g_tild
    sum_g_w += g_tild

wealth = np.exp(np.cumsum(-f))

# Visualization
fig, ax = plt.subplots()

for s in range(n):
    ax.plot(np.arange(T),stocks[s])
ax.plot(np.arange(T), wealth, 'k' , linewidth=5)
plt.show()

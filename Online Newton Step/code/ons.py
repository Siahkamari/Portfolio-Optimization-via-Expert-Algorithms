import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize as op

def project(y, A):
    n = y.size
    theta = np.linalg.solve(A, np.ones(n))
    
    c = np.zeros(4*n)
    A1 = np.concatenate((np.ones([1,n]),np.zeros([1,3*n])), axis = 1)
    A2 = np.concatenate((np.eye(n),-np.eye(n), np.diag(theta), -np.diag(theta)), axis = 1)
    A_eq = np.concatenate((A1, A2), axis = 0)
    b_eq = np.concatenate(([1],y))

    res = op.linprog(c, A_eq = A_eq, b_eq = b_eq, method='revised simplex')

    return  res.x[0:n]

# Data
n = 20
T = 1000

r = 1 + 0.01*np.random.randn(n,T) + 0.001*np.random.randn(n,1) 
stocks =  np.cumprod(r, axis = 1) 

# Parameters
gamma = 20
epsilon = 1/8

# Initialization
A = epsilon*np.eye(n)
x = np.ones(n)/n
f = np.zeros(T)

# Algorithm
for t in range(T-1):
    r = stocks[:,t+1]/stocks[:,t]
    f[t] = - np.log(np.dot(x,r))
    grad = -r/np.dot(x, r)
    A = A + np.outer(grad, grad)
    
    y = x - 1/gamma * np.linalg.solve(A, grad)
    x = project(y, A)

wealth = np.exp(np.cumsum(-f))

# Visualization
fig, ax = plt.subplots()

for s in range(n):
    ax.plot(np.arange(T),stocks[s])
ax.plot(np.arange(T), wealth, 'k' , linewidth=5)
plt.show()

import numpy as np
from sympy import *

x1, x2 = symbols('x1 x2')
x_sym = [x1, x2]
m = len(x_sym)

f1 = 4*x1**2 - 20*x1 + (1/4)*x2**2 + 8
f2 = (1/2)*x1*(x2**2) + 2*x1 - 5*x2 + 8

F = [f1, f2]
n = len(F)

J = [[None for j in range(0,m)] for i in range(0,n)]

for i in range(0,n):
    for j in range(0, m):
        J[i][j] = F[i].diff(x_sym[j])

k = 0
N = 100
tol = 10**(-3)
x = [[0,0]]
y = []


while(k < N+1):

    F_vals = []
    for i in range(0, n):
        F_vals.append(F[i].evalf(subs = {x1: x[k][0], x2: x[k][1]}))

    F_vals = np.array(F_vals, dtype = "float64")
    J_vals = [[None for j in range(0,m)] for i in range(0,n)]

    for i in range(0,n):
        for j in range(0,m):
            J_vals[i][j] = float(J[i][j].evalf(subs = {x1:x[k][0], x2:x[k][1]}))

    y.append(-np.linalg.inv(J_vals)@F_vals)

    x.append(x[k] + y[k])

    if (np.max(np.abs(np.array(y[k]))) < tol):
        print(x[k+1])
        print(k)
        break

    k += 1 

if (k > N):
    print("Maximum number of iterations exceeded.")




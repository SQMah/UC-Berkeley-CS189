import numpy as np
import math

x_1 = np.array([0.2, 3.1, 1])
x_2 = np.array([1.0, 3.0, 1])
x_3 = np.array([-0.2, 1.2, 1])
x_4 = np.array([1.0, 1.1, 1])
X = np.array([x_1, x_2, x_3, x_4])
y = np.array([1, 1, 0, 0])
w = np.array([-1, 1, 0])

if __name__ == "__main__":
    s_vals = np.array([np.dot(x_1, w), np.dot(x_2, w), np.dot(x_3, w), np.dot(x_4, w)])
    print(f"S_0: {s_vals}")
    s = np.array([1/(1+math.e ** -v)for v in s_vals])
    omega = np.array([[s[0]*(1-s[0]), 0, 0, 0], [0, s[1]*(1-s[1]), 0, 0], [0, 0, s[2]*(1-s[2]), 0], [0, 0, 0, s[3]*(1-s[3])]])
    derivative = -np.dot(X.T, y.T - s.T)
    hessian = np.dot(X.T, np.dot(omega, X))
    e = np.linalg.solve(hessian, -derivative)
    w = w + e
    print(f"w_1: {w}")
    s_vals = np.array([np.dot(x_1, w), np.dot(x_2, w), np.dot(x_3, w), np.dot(x_4, w)])
    print(f"S_1: {s_vals}")
    s = np.array([1 / (1 + math.e ** -v) for v in s_vals])
    omega = np.array([[s[0]*(1-s[0]), 0, 0, 0], [0, s[1]*(1-s[1]), 0, 0], [0, 0, s[2]*(1-s[2]), 0], [0, 0, 0, s[3]*(1-s[3])]])
    derivative = -np.dot(X.T, y.T - s.T)
    hessian = np.dot(X.T, np.dot(omega, X))
    e = np.linalg.solve(hessian, -derivative)
    w = w + e
    print(f"w_2: {w}")

import numpy as np
import matplotlib.pyplot as plt

lim = 5
step = 0.1

x, y = np.mgrid[-lim:lim:step, -lim:lim:step]
pos = np.dstack((x, y))


def plot_norm(title, f, ax, p, pos_grid):
    res = []
    for i, chunk in enumerate(pos_grid):
        res.append([])
        for w_1, w_2 in chunk:
            res[i].append((abs(w_1) ** p + abs(w_2) ** p) ** (1 / p))
    contour = ax.contourf(x, y, res, 20)
    f.colorbar(contour)
    ax.axis('square')
    ax.set_title(title)
    ax.set_xlabel("w_1")
    ax.set_ylabel("w_2")
    plt.show()


if __name__ == "__main__":
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_norm("Part a", f, ax, 0.5, pos)
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_norm("Part b", f, ax, 1, pos)
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    plot_norm("Part c", f, ax, 2, pos)

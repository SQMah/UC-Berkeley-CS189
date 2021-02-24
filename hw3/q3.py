import numpy as np
from matplotlib import pylab as plt
from scipy.stats import multivariate_normal


# Isocontour code inspired by Johnathan Betchel's code found here:
# https://jonathonbechtel.com/blog/2018/03/25/adps/

# contours of 2D Gaussian
def plot_isocontours(num, fig, ax, mu, cov, limits, steps):
    lim = limits
    step = steps
    mu_x, mu_y = mu
    x, y = np.mgrid[-lim + mu_x:lim + mu_x:step, -lim + mu_y:lim + mu_y:step]
    rv = multivariate_normal(mu, cov)
    pos = np.dstack((x, y))
    contour = ax.contourf(x, y, rv.pdf(pos), 20)
    fig.colorbar(contour)
    ax.axis('square')
    ax.set_title(f"Subpart {num}")
    # ax.axis('off')


#
def plot_isocontours_subtract(num, fig, ax, mu_1, mu_2, cov_1, cov_2, limits, steps):
    lim = limits
    step = steps
    mu_x_1, mu_y_1 = mu_1
    mu_x_2, mu_y_2 = mu_2
    x, y = np.mgrid[-lim + mu_x_1 - mu_x_2:lim + mu_x_1 - mu_x_2:step, -lim + mu_y_1 - mu_y_2:lim + mu_y_1 - mu_y_2:step]
    rv_1 = multivariate_normal(mu_1, cov_1)
    rv_2 = multivariate_normal(mu_2, cov_2)
    pos = np.dstack((x, y))
    contour = ax.contourf(x, y, rv_1.pdf(pos) - rv_2.pdf(pos), 20)
    fig.colorbar(contour)
    ax.axis('square')
    ax.set_title(f"Subpart {num}")


if __name__ == "__main__":
    # --- SUBPART 1 ---
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    mu = [1, 1]
    cov = np.array([[1, 0],
                    [0, 2]])
    plot_isocontours(1, f, ax, mu, cov, 8, 0.1)

    # --- SUBPART 2 ---
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    mu = [-1, 2]
    cov = np.array([[2, 1],
                    [1, 4]])
    plot_isocontours(2, f, ax, mu, cov, 8, 0.1)

    # --- SUBPART 3 ---
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    mu_1 = [0, 2]
    mu_2 = [2, 0]
    cov_1 = np.array([[2, 1],
                    [1, 1]])
    cov_2 = np.array([[2, 1],
                    [1, 1]])
    plot_isocontours_subtract(3, f, ax, mu_1, mu_2, cov_1, cov_2, 8, 0.1)

    # --- SUBPART 4 ---
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    mu_1 = [0, 2]
    mu_2 = [2, 0]
    cov_1 = np.array([[2, 1],
                      [1, 1]])
    cov_2 = np.array([[2, 1],
                      [1, 4]])
    plot_isocontours_subtract(4, f, ax, mu_1, mu_2, cov_1, cov_2, 8, 0.1)

    # --- SUBPART 5 ---
    f, ax = plt.subplots(1, 1, figsize=(10, 5))
    mu_1 = [1, 1]
    mu_2 = [-1, -1]
    cov_1 = np.array([[2, 0],
                      [0, 1]])
    cov_2 = np.array([[2, 1],
                      [1, 2]])
    plot_isocontours_subtract(5, f, ax, mu_1, mu_2, cov_1, cov_2, 8, 0.1)

    plt.show()

import numpy as np
from matplotlib import pylab as plt

if __name__ == "__main__":
    # Seed np random
    np.random.seed(1234)

    samples = 100

    x_1 = np.random.normal(3, 3, samples)
    x_2_f = np.random.normal(4, 2, samples)
    x_2 = 0.5 * x_1 + x_2_f

    # Compute mean
    print(f"4(a) MEAN: ({np.mean(x_1)}, {np.mean(x_2)})")

    # Compute covariance
    cov = np.cov(np.vstack((x_1, x_2)))
    print(f"4(b) COVARIANCE: {cov}")

    # Compute eifenvectors and eigenvalues
    e_vals, e_vecs = np.linalg.eig(cov)
    print(f"4(c) EIGENVALUES: {e_vals}")
    print(f"4(c) EIGENVECTORS: {e_vecs}")

    # Plot 4(d)
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Plot for 4(d)")
    ax.set_xlabel("R.V. X_1")
    ax.set_ylabel("R.V. X_2")
    ax.scatter(x_1, x_2)
    start_1 = [np.mean(x_1), np.mean(x_1)]
    start_2 = [np.mean(x_2), np.mean(x_2)]
    vec_U = [e_vecs[0][0] * e_vals[0], e_vecs[0][1] * e_vals[1]]
    vec_V = [e_vecs[1][0] * e_vals[0], e_vecs[1][1] * e_vals[1]]
    plt.quiver(start_1, start_2, vec_U, vec_V, angles="xy", scale_units="xy", scale=1)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    plt.show()

    # Plot 4(e)
    x_rotated = np.dot(e_vecs, np.vstack((x_1 - np.mean(x_1), x_2 - np.mean(x_2))))
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title("Plot for 4(e)")
    ax.set_xlabel("R.V. X_1")
    ax.set_ylabel("R.V. X_2")
    ax.scatter(x_rotated[0], x_rotated[1])
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)
    plt.show()

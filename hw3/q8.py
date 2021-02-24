import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import io
import os
import seaborn as sns
from scipy.stats import multivariate_normal
import math

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')

np.random.seed(1234)


def load_mnist():
    return io.loadmat(os.path.join(DATA_DIR, "{}_data.mat".format("mnist")))


def load_spam():
    return io.loadmat(os.path.join(DATA_DIR, "{}_data.mat".format("spam")))


def partition(data, labels, validation_size):
    assert len(data) == len(labels)
    nums = np.arange(len(data))
    np.random.shuffle(nums)
    data_shuffle, labels_shuffle = data[nums], labels[nums]
    return {
        "validation_data": data_shuffle[0: validation_size],
        "validation_labels": labels_shuffle[0: validation_size],
        "training_data": data_shuffle[validation_size:],
        "training_labels": labels_shuffle[validation_size:]
    }


def get_mu_cov(data, labels):
    unique_labels = np.unique(labels)
    mu_cov = {}
    for label in unique_labels:
        idxs = np.array((labels == label)).flatten()
        label_data = data[idxs]
        mu = np.mean(label_data, axis=0)
        cov = np.cov(label_data, rowvar=False)
        mu_cov[label] = {"mu": mu, "cov": cov, "scaled_cov": cov / label_data.shape[0],
                         "prior": label_data.shape[0] / len(labels)}
    return mu_cov


def predict_lda(X, labels, mu_cov, cov_avg):
    preds = [multivariate_normal.logpdf(X, mu_cov[label]["mu"], cov=cov_avg, allow_singular=True)
             + math.log(mu_cov[label]["prior"])
             for label in labels]
    return labels[np.argmax(preds, axis=0)].reshape((-1, 1))


def predict_qda(X, labels, mu_cov):
    preds = [multivariate_normal.logpdf(X, mu_cov[label]["mu"], cov=mu_cov[label]["scaled_cov"], allow_singular=True)
             + math.log(mu_cov[label]["prior"])
             for label in labels]
    return labels[np.argmax(preds, axis=0)].reshape((-1, 1))


def train(model):
    training_points = [100, 200, 500, 1000, 2000, 5000, 10000, 30000, 50000]
    error_rate = []
    per_digit_err = {}
    for i, pts in enumerate(training_points):
        print(f"Processing {model} {pts} training points")
        data_train, data_lab = mnist_t[0: pts], mnist_t_labels[0: pts]
        mu_cov = get_mu_cov(data_train, data_lab)
        covs_sum = sum([mu_cov[label]["cov"] for label in mu_cov])
        cov_avg = covs_sum / float(len(covs_sum))
        if model == "LDA":
            res = predict_lda(mnist_v, np.unique(data_lab), mu_cov, cov_avg)
        elif model == "QDA":
            res = predict_qda(mnist_v, np.unique(data_lab), mu_cov)
        digits = {}
        correct, total = 0, 0
        for i, l in enumerate(mnist_v_labels):
            l = l[0]
            if l not in per_digit_err:
                per_digit_err[l] = []
            if l not in digits:
                digits[l] = (0, 0)
            correct_d, total_d = digits[l]
            if mnist_v_labels[i] == res[i]:
                correct_d += 1
                correct += 1
            total_d += 1
            total += 1
            digits[l] = (correct_d, total_d)
        error_rate.append(1 - correct / total)
        for digit in digits:
            if digit not in per_digit_err:
                per_digit_err[digit] = []
            per_digit_err[digit].append(1 - digits[digit][0] / digits[digit][1])

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"{model} errror rate")
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Training points")
    ax.plot(training_points, error_rate)
    ax.set_ylim((0, 1))
    f.savefig(f"{model}_err_rate.png")

    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_title(f"{model} digitwise error rate")
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Training points")
    for digit in per_digit_err:
        ax.plot(training_points, per_digit_err[digit], label=f"Digit {digit}")
    ax.set_ylim((0, 1))
    ax.legend()
    f.savefig(f"{model}_digit_err_rate.png")


if __name__ == "__main__":
    # Load data
    mnist = load_mnist()
    spam = load_spam()

    # Normalize by l2 norm
    mnist["training_data"] = np.array([mnist["training_data"][i]
                                       / np.linalg.norm(mnist["training_data"][i])
                                       for i in range(len(mnist["training_data"]))])
    mnist_part = partition(mnist["training_data"], mnist["training_labels"], 10000)
    mnist_t = mnist_part["training_data"]
    mnist_t_labels = mnist_part["training_labels"]
    mnist_v = mnist_part["validation_data"]
    mnist_v_labels = mnist_part["validation_labels"]

    # --- Subpart 1 ---
    # Fit guassian to each digit class
    mu_cov = get_mu_cov(mnist_t, mnist_t_labels)

    # --- Subpart 2 ---
    # Visualize covariances
    # Arbitrarily choose the 0th index
    ax = sns.heatmap(
        mu_cov[0]["cov"],
        center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_title(f"Covariance heatmap for digit 0")

    """
    Notice that the on diagonal terms have much stronger positive variance than the off diagonal terms. What this 
    means is that the correlation to nearby pixels (i.e. slightly off diagonal) are stronger than pixels that are
    further away (i.e. further off the diagonal).
    """

    # --- Subpart 3 ---
    # 3(a) - Use LDA
    train(model="LDA")

    # 3(b) - Use QDA
    train(model="QDA")

    # 3(c) Which model performed better? Why?
    """QDA perrformed slightly better as QDA can model more complex quadratic boundaries, as opposed to LDA's
    linear boundaries, which allows QDA to more finly segment different classes. However, QDA also more readily
    overfits."""

    # 3(d) Which digit is easiest to classify?
    """Digit 2 is easiest to classify for QDA, and digit 0 is easiest to classify for LDA."""

    plt.show()

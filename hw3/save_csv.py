# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np
from hw3.q8 import partition, get_mu_cov, predict_qda, predict_lda, load_mnist, load_spam
from scipy import io
import pickle

np.random.seed(1234)

# Usage results_to_csv(clf.predict(X_test))
def results_to_csv(y_test, name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'{name}.csv', index_label='Id')


if __name__ == "__main__":
    # --- Question 8.4 ---
    mnist = load_mnist()
    # Normalize by l2 norm
    mnist["training_data"] = np.array([mnist["training_data"][i]
                                       / np.linalg.norm(mnist["training_data"][i])
                                       for i in range(len(mnist["training_data"]))])
    mnist_part = partition(mnist["training_data"], mnist["training_labels"], 10000)
    mnist_t = mnist_part["training_data"]
    mnist_t_labels = mnist_part["training_labels"]
    mnist_v = mnist_part["validation_data"]
    mnist_v_labels = mnist_part["validation_labels"]
    data_train, data_lab = mnist_t[0: 50000], mnist_t_labels[0: 50000]
    mu_cov = get_mu_cov(data_train, data_lab)
    res = predict_qda(np.array([mnist["test_data"][i]
                                / np.linalg.norm(mnist["test_data"][i])
                                for i in range(len(mnist["test_data"]))]), np.unique(mnist_t_labels),
                      mu_cov=mu_cov).flatten()
    results_to_csv(res, "mnist")

    # --- Question 8.5 ---
    spam = load_spam()
    mu_cov = get_mu_cov(spam["training_data"], spam["training_labels"])
    covs_sum = sum([mu_cov[label]["cov"] for label in mu_cov])
    cov_avg = covs_sum / float(len(covs_sum))
    res = predict_lda(spam["test_data"], np.unique(spam["training_labels"]), mu_cov=mu_cov, cov_avg=cov_avg).flatten()
    results_to_csv(res, "spam")

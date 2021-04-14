import numpy as np
import os
from scipy import io
from scipy.special import expit as s
import matplotlib.pyplot as plt
import pandas as pd

# --- RANDOM SEED ---
np.random.seed(1234)

# --- Parameters ---
lamb = 1

CURR_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(CURR_DIR, 'data')


def load_wine():
    return io.loadmat(os.path.join(DATA_DIR, "data.mat"))


wine = load_wine()


def normalize(data):
    data -= np.tile(np.mean(data, axis=0), (data.shape[0], 1))
    data = np.divide(data, np.tile(np.std(data, axis=0),
                                   (data.shape[0], 1)))
    data = np.hstack((data, np.ones((data.shape[0], 1))))
    return data


def partition(data, labels, validation_size):
    assert len(data) == len(labels)
    nums = np.arange(len(data))
    np.random.shuffle(nums)
    data_shuffle, labels_shuffle = data[nums], labels[nums]
    res = {
        "validation_data": data_shuffle[0: validation_size],
        "validation_labels": labels_shuffle[0: validation_size],
        "training_data": data_shuffle[validation_size:],
        "training_labels": labels_shuffle[validation_size:]
    }
    res["training_data"] = normalize(res["training_data"])
    res["validation_data"] = normalize(res["validation_data"])
    return res


part = partition(wine["X"], wine["y"], int(len(wine["X"]) * 0.2))

X = part["training_data"]
y = part["training_labels"]
X_v = part["validation_data"]
y_v = part["validation_labels"]


def calc_cost(w):
    cost_X = X_v
    cost_y = y_v
    return sum(
        [-cost_y[i] * np.log(s(np.dot(cost_X[i], w))) - (1 - cost_y[i]) * np.log(1 - s(np.dot(cost_X[i], w))) for i in
         range(X_v.shape[0])]) + lamb * np.linalg.norm(w, ord=2)
    # s_res = s(np.dot(X_v, w))
    # print(s_res)
    # s_0 = np.array(list(map(np.log, s_res)))
    # s_1 = np.array(list(map(lambda x: 1 - np.log(x), s_res)))
    # return (-np.dot(y_v.T, s_0) - np.dot((1 - y_v).T, s_1) + lamb * np.linalg.norm(w, ord=2))[0][0]


def plot_costs(title, costs, iters):
    f, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.plot(iters, costs)
    ax.set_title(title)
    ax.set_xlabel("Number of iterations")
    ax.set_ylabel("Cost")
    plt.savefig(f"{title}.png")
    plt.show()


def batch_gradient_descent(w):
    lr = 0.0001
    iters_to_plot = np.arange(0, 20500, 500)
    print("===Batch Gradient Descent===")
    costs = []
    min_w = w
    for i in range(0, max(iters_to_plot) + 1):
        w = w + lr * (np.dot(X.T, (y - s(np.dot(X, w)))) + 2 * lamb * w)
        if i in iters_to_plot:
            print(f"Iteration {i}")
            cost = calc_cost(w)
            costs.append(cost)
            if cost == min(costs):
                min_w = w
    plot_costs("Batch Gradient Descent Costs", costs, iters_to_plot)
    return min_w, min(costs)


def stochastic_gradient_descent(w):
    lr = 0.001
    iters_to_plot = np.arange(0, 20500, 500)
    print("===Stochastic Gradient Descent===")
    costs = []
    min_w = w
    for i in range(0, max(iters_to_plot) + 1):
        w = w + lr * ((y[i % y.size][0] - s(np.dot(X[i % X.shape[0]], w)))[0] * X[i % X.shape[0]].reshape(w.shape)
                      + 2 * lamb * w)
        if i in iters_to_plot:
            print(f"Iteration {i}")
            cost = calc_cost(w)
            costs.append(cost)
            if cost == min(costs):
                min_w = w
    plot_costs("Stochastic Gradient Descent Costs", costs, iters_to_plot)
    return min_w, min(costs)


def annealing_stochastic_gradient_descent(w):
    sigma = 0.1
    iters_to_plot = np.arange(0, 100500, 500)
    print("===Annealing Stochastic Gradient Descent===")
    costs = []
    min_w = w
    for i in range(0, max(iters_to_plot) + 1):
        w = w + (sigma / (i + 1)) * ((y[i % y.size][0] - s(np.dot(X[i % X.shape[0]], w)))[0] *
                                     X[i % X.shape[0]].reshape(w.shape)
                                     + 2 * lamb * w)
        if i in iters_to_plot:
            print(f"Iteration {i}")
            cost = calc_cost(w)
            costs.append(cost)
            if cost == min(costs):
                min_w = w
    plot_costs("Annealing Stochastic Gradient Descent Costs", costs, iters_to_plot)
    return min_w, min(costs)


def inference(X, w):
    prob = np.dot(X, w)

    def helper(p):
        p = p[0]
        if s(p) >= 0.5:
            return 1
        else:
            return 0

    return np.array(list(map(helper, prob)))


def results_to_csv(y_test, name):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv(f'{name}.csv', index_label='Id')


if __name__ == "__main__":
    init_w = np.zeros((X.shape[1], 1))
    b_w, b_w_cost = batch_gradient_descent(init_w)
    s_w, s_w_cost = stochastic_gradient_descent(init_w)
    a_w, a_w_cost = annealing_stochastic_gradient_descent(init_w)
    idx = np.argmin([b_w_cost, s_w_cost, a_w_cost])
    print(f"Best model at index {idx}")
    best_w = [b_w, s_w, a_w][idx]
    res = inference(X_v, best_w)
    print(y_v, res)
    results_to_csv(inference(normalize(wine["X_test"]), best_w), "wine_results")

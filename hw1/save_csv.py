# A code snippet to help you save your results into a kaggle accepted csv
import pandas as pd
import numpy as np
from hw1 import load_data
from scipy import io
import pickle


# Usage results_to_csv(clf.predict(X_test))
def results_to_csv(y_test):
    y_test = y_test.astype(int)
    df = pd.DataFrame({'Category': y_test})
    df.index += 1  # Ensures that the index starts at 1.
    df.to_csv('submission.csv', index_label='Id')


if __name__ == "__main__":
    data = io.loadmat("data/cifar10_data.mat")
    clf = pickle.load(open("cifar/CIFAR.model", "rb"))
    print(data['training_data'].shape)
    print(data['test_data'].shape)
    y_test = clf.predict(data["test_data"])
    results_to_csv(y_test)

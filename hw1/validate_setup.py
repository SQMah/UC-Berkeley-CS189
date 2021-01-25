if __name__ == "__main__":
    import sys

    if sys.version_info[0] < 3:
        raise Exception("Python 3 not detected.")
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn import svm
    from scipy import io

    for data_name in ["mnist", "spam", "cifar10"]:
        data = io.loadmat("data/%s_data.mat" % data_name)
        print("\nloaded %s data!" % data_name)
        fields = "test_data", "training_data", "training_labels"
        for field in fields:
            print(field, data[field].shape)

    mnist = io.loadmat("data/mnist_data.mat")
    print(mnist["training_data"][0:10], mnist["training_labels"][0:10])
    nums = np.arange(len(mnist["training_data"]))
    np.random.shuffle(nums)
    dat = mnist["training_labels"]
    print("SHAPE", dat.T[0], dat.T[0].shape)

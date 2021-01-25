"""
Author: Joey Hejna
Institution: UC Berkeley
Date: Spring 2021
Course: CS189/289A

A Template file for CS 189 Homework 1.

Feel free to use this if you like, but you are not required to!
"""

# Imports. Feel free to add / remove
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score  # This is the only function you are allowed to use.
from scipy import io
import csv
import os
import argparse


#################################
# Suggested Utility Functions   #
#################################

def load_data(name):
    if name in {"mnist", "spam", "cifar10"}:
        data = io.loadmat("data/%s_data.mat" % name)
        return data
    raise ValueError(f"Dataset {name} is not defined!")


#################################
# Question 1: Data Partitioning #
#################################

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


def main_q1():
    mnist_raw = load_data("mnist")
    spam_raw = load_data("spam")
    cifar10_raw = load_data("cifar10")

    # Set aside 10,000 validation for MNIST
    mnist_part = partition(mnist_raw["training_data"], mnist_raw["training_labels"], 50000)

    # Set aside 20% of the data for validation for spam
    spam_part = partition(spam_raw["training_data"], spam_raw["training_labels"], int(len(spam_raw["training_data"]) *
                                                                                      0.2))

    # Set aside 5000 images for CIFAR
    cifar10_part = partition(cifar10_raw["training_data"], cifar10_raw["training_labels"], 5000)

    return mnist_part, spam_part, cifar10_part


#################################
# Question 2: SVMs              #
#################################

def train(X, Y, **kwargs):
    # create, train, and return and SVM Model
    clf = svm.LinearSVC(max_iter=50000, **kwargs)
    clf.fit(X, Y)
    return clf


def num_examples_experiment(plot_name, fig_num, X_train, Y_train, X_val, Y_val, num_examples_arr, **kwargs):
    # train an svm for each number of examples.
    # Evaluate the training and validation performance
    # plot the results.
    examples_arr = []
    train_acc_arr, val_acc_arr = [], []

    for num in num_examples_arr:
        print(f"{plot_name} {num}...")
        if num == "ALL":
            num = len(X_train) - 1
        X_trainT = X_train[0: num]
        Y_trainT = Y_train.T[0][0: num]
        clf = train(X_trainT, Y_trainT, **kwargs)
        train_acc = accuracy_score(clf.predict(X_trainT), Y_trainT)
        val_acc = accuracy_score(clf.predict(X_val), Y_val.T[0])
        examples_arr.append(num)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
    plt.figure(fig_num)
    plt.plot(examples_arr, train_acc_arr, label="Training accuracy")
    plt.plot(examples_arr, val_acc_arr, label="Validation accuracy")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title(plot_name)
    plt.savefig(plot_name + ".pdf")


def main_q2():
    # Run all of the code for question 3.
    MNIST_NUM_EXAMPLES_ARR = [100, 200, 500, 1000, 2000, 5000, 10000]
    SPAM_NUM_EXAMPLES_ARR = [100, 200, 500, 1000, 2000,
                             "ALL"]  # Figure out the number of examples in the dataset or handle this case.
    CIFAR_NUM_EXAMPLES_ARR = [100, 200, 500, 1000, 2000, 5000]

    ALL_ARR = [MNIST_NUM_EXAMPLES_ARR, SPAM_NUM_EXAMPLES_ARR, CIFAR_NUM_EXAMPLES_ARR]
    PLT_NAMES = ["mnist", "spam", "cifar10"]
    PARTITIONS = main_q1()
    ARGS = [{}, {}, {}]
    for i, arr in enumerate(ALL_ARR):
        curr = PARTITIONS[i]
        num_examples_experiment(PLT_NAMES[i], i, curr["training_data"], curr["training_labels"],
                                curr["validation_data"], curr["validation_labels"], ALL_ARR[i])
    plt.show()


#################################
# Question 3: Hyperparameters   #
#################################

def hyperparameter_search(X, Y, X_val, Y_val, n, C):
    print(f"Testing with c value: {C}")
    if n == "ALL":
        n = len(X) - 1
    X_trainT = X[0: n]
    Y_trainT = Y.T[0][0: n]
    clf = train(X_trainT, Y_trainT, C=C)
    train_acc = accuracy_score(clf.predict(X_trainT), Y_trainT)
    val_acc = accuracy_score(clf.predict(X_val), Y_val.T[0])
    print(f"TRAIN ACCURACY: {train_acc}, VALIDATION ACCURACY: {val_acc}")
    return train_acc, val_acc


def main_q3():
    MNIST_NUM_EXAMPLES = 10000
    PARTITIONS = main_q1()
    curr = PARTITIONS[0]
    c_arr = [100, 10, 1.0, 0.1, 0.01]
    train_acc_arr, val_acc_arr = [], []
    for c in c_arr:
        train_acc, val_acc = hyperparameter_search(curr["training_data"],
                                                   curr["training_labels"], curr["validation_data"],
                                                   curr["validation_labels"], MNIST_NUM_EXAMPLES, c)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
    plt.figure(1)
    plt.plot(c_arr, train_acc_arr, label="Training accuracy")
    plt.plot(c_arr, val_acc_arr, label="Validation accuracy")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.title("Coarse search for C values for MNIST.")
    plt.legend()
    plt.savefig("Coarse MNIST C.pdf")

    print("========")
    # From the coarse values, choose the best one, and then search around it.
    max_acc = max(val_acc_arr)
    max_i = 0
    for i, acc in enumerate(val_acc_arr):
        if acc == max_acc:
            max_i = i
    best_c = c_arr[max_i]
    print(f"BEST C: {best_c}")
    if max_i != 0 and max_i != len(c_arr) - 1:
        diff_l = (c_arr[max_i - 1] - c_arr[max_i]) / 4
        diff_r = (c_arr[max_i] - c_arr[max_i + 1]) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff_l)
            to_check.append(best_c - i * diff_r)
    elif max_i == 0:
        diff = (best_c - c_arr[1]) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff)
            to_check.append(best_c - i * diff)
    else:
        diff = (c_arr[-2] - best_c) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff)
            to_check.append(best_c - i * diff)

    train_acc_arr = []
    val_acc_arr = []
    to_check = [v for v in to_check if v >= 0]
    for c in to_check:
        train_acc, val_acc = hyperparameter_search(curr["training_data"],
                                                   curr["training_labels"], curr["validation_data"],
                                                   curr["validation_labels"], MNIST_NUM_EXAMPLES, c)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
    plt.figure(2)
    plt.plot(to_check, train_acc_arr, label="Training accuracy")
    plt.plot(to_check, val_acc_arr, label="Validation accuracy")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.title("Finer search for C values for MNIST.")
    plt.legend()
    plt.savefig("Fine MNIST C.pdf")

    plt.show()


#################################
# Question 4: KFold CrossValid  #
#################################

def k_fold_cross_validation(X_train, Y_train, k, n, C):
    # X_train, Y_train is already shuffled on partition.
    print(f"Training model with C={C}")
    if n == "ALL":
        n = len(X_train) - 1
    X_train = X_train[0: n]
    Y_train = Y_train.T[0][0: n]
    part_size = int(n / k)
    train_arr = []
    val_arr = []
    for i in range(0, k):
        print(f"Partition {i}")
        if i == k - 1:
            valid_X = X_train[(k - 1) * part_size:]
            valid_Y = Y_train[(k - 1) * part_size:]
            cut_X = X_train[0: (k - 1) * part_size]
            cut_Y = Y_train[0: (k - 1) * part_size]
        else:
            valid_X = X_train[i * part_size: (i + 1) * part_size]
            valid_Y = Y_train[i * part_size: (i + 1) * part_size]
            cut_X = X_train[0: (i - 1) * part_size] + X_train[(i + 1) * part_size:]
            cut_Y = Y_train[0: (i - 1) * part_size] + Y_train[(i + 1) * part_size:]
        clf = train(cut_X, cut_Y, C=C)
        train_acc = accuracy_score(clf.predict(cut_X), cut_Y)
        val_acc = accuracy_score(clf.predict(valid_X), valid_Y)
        train_arr.append(train_acc)
        val_arr.append(val_acc)
        print(f"TRAIN ACCURACY: {train_acc}, VALIDATION ACCURACY: {val_acc}")
    print(f"AVG TRAIN: {sum(train_arr) / len(train_arr)}, AVG VAL: {sum(val_arr) / len(val_arr)}")
    return sum(train_arr) / len(train_arr), sum(val_arr) / len(val_arr)


def main_q4():
    curr = main_q1()[1]
    c_arr = [100, 10, 1.0, 0.1, 0.01]
    train_acc_arr, val_acc_arr = [], []
    for c in c_arr:
        train_acc, val_acc = k_fold_cross_validation(curr["training_data"],
                                                     curr["training_labels"], 5, len(curr["training_data"]),
                                                     C=c)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
    plt.figure(1)
    plt.plot(c_arr, train_acc_arr, label="Training accuracy")
    plt.plot(c_arr, val_acc_arr, label="Validation accuracy")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.title("Coarse search for C values for MNIST.")
    plt.legend()
    plt.savefig("Coarse spam C.pdf")

    print("========")

    # From the coarse values, choose the best one, and then search around it.
    max_acc = max(val_acc_arr)
    max_i = 0
    for i, acc in enumerate(val_acc_arr):
        if acc == max_acc:
            max_i = i
    best_c = c_arr[max_i]
    print(f"BEST C: {best_c}")
    if max_i != 0 and max_i != len(c_arr) - 1:
        diff_l = (c_arr[max_i - 1] - c_arr[max_i]) / 4
        diff_r = (c_arr[max_i] - c_arr[max_i + 1]) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff_l)
            to_check.append(best_c - i * diff_r)
    elif max_i == 0:
        diff = (best_c - c_arr[1]) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff)
            to_check.append(best_c - i * diff)
    else:
        diff = (c_arr[-2] - best_c) / 4
        to_check = []
        for i in range(1, 4):
            to_check.append(best_c + i * diff)
            to_check.append(best_c - i * diff)

    train_acc_arr = []
    val_acc_arr = []
    to_check = [v for v in to_check if v >= 0]
    for c in to_check:
        train_acc, val_acc = k_fold_cross_validation(curr["training_data"],
                                                     curr["training_labels"], 5, len(curr["training_data"]),
                                                     C=c)
        train_acc_arr.append(train_acc)
        val_acc_arr.append(val_acc)
    plt.figure(2)
    plt.plot(to_check, train_acc_arr, label="Training accuracy")
    plt.plot(to_check, val_acc_arr, label="Validation accuracy")
    plt.xlabel("Number of examples")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    plt.title("Finer search for C values for spam.")
    plt.legend()
    plt.savefig("Fine spam C.pdf")

    plt.show()


#################################
# Question 5: Kaggle            #
#################################

# For kaggle, do whatever you like!

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", "-q", type=int, default=1, help="Specify which question to run")
    args = parser.parse_args()

    if args.question == 1:
        main_q1()
    if args.question == 2:
        main_q2()
    elif args.question == 3:
        main_q3()
    elif args.question == 4:
        main_q4()
    elif args.question == 5:
        pass  # TODO: Insert your calls for running the kaggle code.
    else:
        raise ValueError("Cannot find specified question number")

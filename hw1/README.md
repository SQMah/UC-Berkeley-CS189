#Reproduction steps:
- For MNIST:

Change the svm in the train function to SVM instead of LinearSVM, and add the argument: kernel='rbf' to the SVM, also change number of iterations to 1000.
Run question 3, which will print the best C value. Take the saved .model file corresponding to that c value in the directory /mnist/MNIST_{C}.model, and use save_csv.py to get the results.

- For CIFAR10:

Change the svm in the train function to SVM instead of LinearSVM, and add the argument: kernel='rbf' to the SVM, also change number of iterations to 1000. Run question 2, and
take the saved .model file in the directory /cifar/CIFAR.model, and use save_csv.py to get results.

- For SPAM:

Run question 3, which will print the best C value. Take the saved .model file corresponding to that c value in the directory /spam/SPAM_{C}.model, and use save_csv.py to get the results.

import mysklearn.myutils as myutils
import numpy as np
import math
import random
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of sample_sizes
            The shape of X is (n_sample_sizes, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_sample_sizes
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training sample_sizes
        X_test(list of list of obj): The list of testing sample_sizes
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       np.random.seed(0) 
       pass
  
    if shuffle:
       # TODO: shuffle the rows in X and y before splitting
       # be sure to maintain the parallel order of X and y!!
       # note: the unit test for train_test_split() does not test
       # your use of random_state or shuffle, but you should still
       # implement this and check your work yourself
       for i in range(len(X)):
           rand_index = random.randrange(0, len(X)) # [0, len(X))
           X[i], X[rand_index] = X[rand_index], X[i] # this is the temporary value swap but in one line
           if y is not None:
               y[i], y[rand_index] = y[rand_index], y[i]
       pass
    
    num_instances = len(X) # 8
    if isinstance(test_size, float):
        test_size = math.ceil(num_instances * test_size) # ceil(8 * 0.33)
    split_index = num_instances - test_size # 8 - 2 = 6

    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=5):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of sample_sizes
            The shape of X is (n_sample_sizes, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_sample_sizes % n_splits folds have size n_sample_sizes // n_splits + 1, 
            other folds have size n_sample_sizes // n_splits, where n_sample_sizes is the number of sample_sizes
            (e.g. 11 sample_sizes and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 sample_sizes)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    
    1. make fold list
    2. loop through fold list and 
        loop through x and 
        grab current x, put into current fold
        keep going until no more x
    y = class labels ie "yes" or "no" 1 or 0 
    
    test will loop though folds
        test current fold
        train on the other folds
            keep track of predictions
    lecture 3/2 42:00
    """
    #calc number of folds needed
    n = len(X)
    num_folds = n % n_splits
    #initiate train and test
    X_train_folds = []
    X_test_folds = []
    sample_size = []

    for i in range(n_splits):
        if i >= num_folds:
            sample_size.append(n // n_splits)
        else:
            sample_size.append((n // n_splits) + 1)

    for i in range(n_splits):
        indices = [j for  j in range(len(X))]
        range_size = sample_size[i]
        start_index = sum(sample_size[n] for n in range(i))
        test_fold = [k for k in range(start_index, start_index + range_size)]
        X_test_folds.append(test_fold)
        del indices[start_index:start_index + range_size]
        X_train_folds.append(indices)
 
    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (sample_sizes). 
            The shape of X is (n_sample_sizes, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_sample_sizes
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    
    1. make fold list
    2. group by class label
    3. loop through each class label group
        loop through folds
        add every instance from group to every fold
    """
    total_folds = [[] for _ in range(n_splits)]
    X_train_folds = [[] for _ in range(n_splits)]
    X_test_folds = [[] for _ in range(n_splits)]

    #group by
    y_list, counts = myutils.get_freq_1col(y)
    group_subtables = [[] for _ in y_list]
    for i, val in enumerate(y):
        for j, label in enumerate(y_list):
            if val == label:
                group_subtables[j].append(i)
 
    curr = 0
    for group in group_subtables:
        for i in group:
            total_folds[curr].append(i)
            curr = (curr + 1) % n_splits
    curr = 0
    for j in range(n_splits):
        for i, fold in enumerate(total_folds):
            if(i != j):
                for val in fold:
                    X_train_folds[curr].append(val)
            else:
                X_test_folds[curr] = fold
        curr += 1
 
    return X_train_folds, X_test_folds


def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_sample_sizes
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_sample_sizes
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of sample_sizes with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    matrix = []
    for _ in range(len(labels)):
        matrix_row = []
        for _ in range(len(labels)):
            matrix_row.append(0)
        matrix.append(matrix_row)
 
    for i in range(len(labels)):
        label = labels[i]
        for j in range(len(y_true)):
            if y_true[j] == label:
                index = labels.index(y_pred[j])
                matrix[i][index] = matrix[i][index] + 1
    return matrix


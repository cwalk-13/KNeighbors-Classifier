import mysklearn.myutils as myutils
import numpy as np
import math
import random
import copy

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
       # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
       np.random.seed(random_state)
       pass
    
    if shuffle: 
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        for i in range(len(X)):
            
            rand_index = random.randrange(len(X))
            #np.random.uniform(0, len(X), len(X)) # [0, len(X))
            X[i], X[rand_index] = X[rand_index], X[i]
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
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
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
    folds = []
    for i in range(n_splits):
        fold = []
        folds.append(fold)
    fold_index = 0
    # for fold in folds:
    for i in range(len(X)):

        if fold_index == n_splits:
            fold_index = 0
        folds[fold_index].append(i)
        # print("this is fold: ", fold_index)
        # print(fold) 
        fold_index += 1

    X_train_folds = folds
    folds_copy = copy.copy(folds)
   
    # folds_copy.reverse()
    X_test_folds = []

    # for fold in folds:
    #     X_test_folds.append(fold)
    
    for fold in folds_copy[::-1]:
        X_test_folds.append(fold)

    
    
    # print(folds)
    # print(X_train_folds)
    # print(X_test_folds)
    return X_train_folds, X_test_folds # TODO: fix this
    # return [[1, 3], [0, 2]], [[0, 2], [1, 3]] # TODO: fix this

def stratified_kfold_cross_validation(X, y, n_splits=5):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
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
    print("called")
    folds = []
    for i in range(n_splits):
        fold = []
        folds.append(fold)
    
    #add  y to x
    for i, instance in enumerate(X):
        # append the class label
        instance.append(y[i])
        instance.append(i)
    #groupby
    group_names = sorted(list(set(y))) 
    group_subtables = [[] for _ in group_names]
    for row in X:
        group_by_value = row[-2]
        group_index = group_names.index(group_by_value)
        index = row.pop()
        group_subtables[group_index].append(index) # shallow copy
    
    #add every instance to every fold
    for group in group_subtables:
        fold_index = 0
        for i in range(len(group)):
            if fold_index == n_splits:
                fold_index = 0
            folds[fold_index].append(group[i])
            fold_index += 1
   
    folds_copy = folds
    test_folds = []
    for i in folds_copy:
        test_folds.insert(0, i)
    X_train_folds = folds
    X_test_folds = folds
    print(X_test_folds)
    print(X_train_folds)
    print(test_folds)


    return X_train_folds, X_test_folds # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    return [] # TODO: fix this

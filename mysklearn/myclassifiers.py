import mysklearn.myutils as myutils
import numpy as np
import copy
import operator

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        x = []
        for sample in X_train:
            x.append(sample[0])

        y = y_train
        mean_x = np.mean(x)
        mean_y = np.mean(y)
        
        m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
        b = mean_y - m * mean_x 
        self.intercept = b
        self.slope = m 
  
        pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        x = []
        for sample in X_test:
            x.append(sample[0])
        
        y = []
        for i in range(len(x)):
            y_val = round((self.slope * x[i]) + self.intercept, 5)
            y.append(y_val)

        return y # TODO: fix this


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train 
        self.y_train = y_train 


    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        #enumerate returns pairs, first element is index second element is elememnt
        #from knn example in CLassificationFun
        train = copy.deepcopy(self.X_train)
        k = self.n_neighbors

        all_distances = []
        all_neighbor_indices = []
        for test in X_test:
            for i, instance in enumerate(train):
                # append the class label
                instance.append(self.y_train[i])
                # append the original row index
                instance.append(i)
                # append the distance to [2, 3]
                dist = myutils.compute_euclidean_distance(instance[:2], test)
                instance.append(dist)
            
            # sort train by distance
            train_sorted = sorted(train, key=operator.itemgetter(-1))

            # grab the top k
            top_k = train_sorted[:k]
            dists = []
            indices = []
            for instance in top_k:
                dists.append(instance[-1])
                indices.append(instance[-2])
            # print("Top K Neighbors")
            # for instance in top_k:
            #     print(instance)
            all_distances.append(dists)
            all_neighbor_indices.append(indices)
        
        return all_distances, all_neighbor_indices # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        dists, all_indices = self.kneighbors(X_test)

        predicted_y_vals = []
        for indices in all_indices:
            y_vals = []
            for index in indices:
                y_vals.append(self.y_train[index])
            values, counts = myutils.get_freq_1col(y_vals)

            index_avg, avg = max(enumerate(counts), key=operator.itemgetter(1))
            print(index_avg)
            predicted_y_vals.append(values[index_avg])
        print(predicted_y_vals)
        return predicted_y_vals # TODO: fix this

import numpy as np
import scipy.stats as stats
import mysklearn.myevaluation as myeval

from mysklearn.myclassifiers import MySimpleLinearRegressor, MyKNeighborsClassifier

# note: order is actual/received student value, expected/solution
def test_simple_linear_regressor_fit():
    #x, y dataset 1
    np.random.seed(0)
    x1 = list(range(0, 100))
    y1 = [value * 2 + np.random.normal(0, 25) for value in x1]
    sp_m, sp_b, sp_r, sp_r_p_val, sp_std_err = stats.linregress(x1, y1)
    reg = MySimpleLinearRegressor()
    x = []
    for i in range(len(x1)):
        sample = [x1[i]]
        x.append(sample)
    reg.fit(x, y1)

    #x,y dataset2
    y2 = [value ** 2 + np.random.normal(0, 25) for value in x1]
    sp_m2, sp_b2, sp_r2, sp_r_p_val2, sp_std_err2 = stats.linregress(x1, y2)
    reg2 = MySimpleLinearRegressor()
    reg2.fit(x, y2)
    assert np.isclose(reg.slope, sp_m), np.isclose(reg.intercept, sp_b) 
    #test dataset2
    assert np.isclose(reg2.slope, sp_m2), np.isclose(reg2.intercept, sp_b2)

def test_simple_linear_regressor_predict():
    np.random.seed(0)
    x1 = list(range(0, 100))
    x = []
    for i in range(len(x1)):
        sample = [x1[i]]
        x.append(sample)
    
    y1 = [value * 2 for value in x1]
    reg = MySimpleLinearRegressor()
    reg.fit(x, y1)
    x1_test = x[0:100:3]
    y1_vals = y1[0:100:3]
    y1_predict = reg.predict(x1_test)
    assert np.allclose(y1_vals, y1_predict)

    y2 = [value * 0.5 + 5 for value in x1]
    reg2 = MySimpleLinearRegressor()
    reg2.fit(x, y2)
    x1_test = x[0:100:3]
    y2_vals = y2[0:100:3]
    y2_predict = reg2.predict(x1_test)
    assert np.allclose(y2_vals, y2_predict)

def test_kneighbors_classifier_kneighbors():
    train0 = [
        [7, 7],
        [7, 4],
        [3, 4],
        [1, 4]
    ]
    train0_labels = ["bad", "bad", "good", "good"]
    test0 = [[3, 7]]
    knn0 = MyKNeighborsClassifier()
    knn0.fit(train0, train0_labels)
    dists0, indices0 = knn0.kneighbors(test0)
    real_indices = [[2, 3, 0]]
    assert indices0 == real_indices
    train1 = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train1_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test1 = [[2, 3]]
    knn1 = MyKNeighborsClassifier()
    knn1.fit(train1, train1_labels)
    dists1, indices1 = knn1.kneighbors(test1)
    real_indices = [[0, 4, 6]]

    assert indices1 == real_indices # TODO: fix this

def test_kneighbors_classifier_predict():
    train0 = [
        [7, 7],
        [7, 4],
        [3, 4],
        [1, 4]
    ]
    train0_labels = ["bad", "bad", "good", "good"]
    test0 = [[3, 7]]
    knn0 = MyKNeighborsClassifier()
    knn0.fit(train0, train0_labels)
    predicted0 = knn0.predict(test0)
    actual = ["good"] 
    assert predicted0 == actual
   
    train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    train_labels = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    test = [[2, 3]]
    knn1 = MyKNeighborsClassifier()
    knn1.fit(train, train_labels)
    predicted = knn1.predict(test)
    actual = ["yes"] 
    assert predicted == actual # TODO: fix this

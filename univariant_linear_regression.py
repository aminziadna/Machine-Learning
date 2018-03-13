import sqlite3

# load and prepare data
connection = sqlite3.connect("C:\\Users\\check\\Downloads\\soccer_database.sqlite")
cur = connection.cursor()
cur.execute("select height,weight from Player")
myData = cur.fetchall()
cur.close()

from random import seed
from random import randrange
from math import sqrt
import matplotlib.pyplot as plt
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy
def rmse_metric(actual, predicted):
    sum_error = 0.0
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error ** 2)
    mean_error = sum_error / float(len(actual))
    return sqrt(mean_error)


# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse, actual, predicted, test

def mean(values):
    return sum(values) / float(len(values))


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])


# Calculate coefficients
def coefficients(dataset):
    x = [row[0] for row in dataset]
    y = [row[1] for row in dataset]
    x_mean, y_mean = mean(x), mean(y)
    b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
    b0 = y_mean - b1 * x_mean
    return [b0, b1]


# Simple linear regression algorithm
def simple_linear_regression(train, test):
    predictions = list()
    b0, b1 = coefficients(train)
    for row in test:
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)
    return predictions


# Simple linear regression on dataset
seed(1)
# evaluate algorithm
split = 0.6
rmse, yActual, yPredicted, xValues = evaluate_algorithm(myData, simple_linear_regression, split)
print 'RMSE: %.3f' % (rmse)
fig = plt.figure()

ax1 = fig.add_subplot(111)

ax1.set_title("Univariant Linear Regression")
ax1.set_xlabel('Width (cm)')
ax1.set_ylabel('Height (cm)')
val = [x[0] for x in xValues]
ax1.plot(val, yPredicted, c='r', label='predicted')
ax1.scatter(val, yActual, c='b', label='real')
leg = ax1.legend()

plt.show()

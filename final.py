#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
import seaborn as sns

# Import the datasets
from sklearn import datasets

def load_boston():
    boston = datasets.load_boston()
    #print(boston.DESCR)
    x = boston.data
    y = boston.target.reshape([len(boston.target), 1])
    return x, y

# :Attribute Information (in order):
#  0    - CRIM     per capita crime rate by town
#  1    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#  2    - INDUS    proportion of non-retail business acres per town
#  3    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#  4    - NOX      nitric oxides concentration (parts per 10 million)
#  5    - RM       average number of rooms per dwelling
#  6    - AGE      proportion of owner-occupied units built prior to 1940
#  7    - DIS      weighted distances to five Boston employment centres
#  8    - RAD      index of accessibility to radial highways
#  9    - TAX      full-value property-tax rate per $10,000
# 10    - PTRATIO  pupil-teacher ratio by town
# 11    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 12    - LSTAT    % lower status of the population
# 13    - MEDV     Median value of owner-occupied homes in $1000's


def normalise(x, mu=None, sigma=None):
    """Normalises a matrix."""
    if mu is None and sigma is None:
        mu = np.mean(x, axis=0)
        sigma = np.std(x, axis=0)
        return (x - mu) / sigma, mu, sigma
    else:
        return (x - mu) / sigma, mu, sigma


def add_theta_intercept(x):
    """Prepends a column of all ones to x."""
    theta0_column = np.ones(len(x))
    x = np.matrix( np.c_[theta0_column, x] )
    return x

def regression_cost(x, y, theta):
    """Linear regression cost function."""
    m = len(y)
    h = x * theta
    z = h - y
    j = (2 * m)**-1 * float(np.dot(z.T, z))
    return j

def regression_train(x, y, theta, iterations, alpha):
    """Linear regression training."""
    m = len(y)
    history = []
    for i in range(iterations):
        h = x * theta
        z = h - y
        theta = theta - alpha * m**-1 * (x.T * z)
        history.append(regression_cost(x, y, theta))
    return theta, np.array(history)

def sumsqr(a, b):
    c = a - b
    return np.dot(c.T, c)

def rsquared(y_observed, y_predicted):
    """R^2 is the fraction by which the variance of the errors is less than
    the variance of the dependent variable."""
    mean = np.mean(y_observed)
    ss_tot = sumsqr(y_observed, mean)
    ss_res = sumsqr(y_observed, y_predicted)
    return 1.0 - float(ss_res / ss_tot)


class LinearRegression(object):
    def __init__(self, features, target, normalised = False):
        if normalised:
            self.x, self.mu, self.sigma = normalise(features)
        else:
            self.x = features
            self.mu = None
            self.sigma = None
        self.x = add_theta_intercept(self.x)
        self.y = target
        self.theta = np.matrix(np.zeros([self.x.shape[1], 1]))
        self.normalised = normalised
        self.cost_history = np.array([])

    def plot(self, dim=0, xlabel='', ylabel='', filename=None):
        plt.plot(self.x[:,1+dim], self.y, "bo")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        xval = self.x[:,1+dim]
        yval = self.x * self.theta
        plt.plot(xval, yval, '-')
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

        plt.close()

    def plot_j(self):
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost J')
        #plt.show()
        plt.savefig('cost_function.png')
        plt.close()


    def cost(self):
        """Computes the cost function based on the current values."""
        return regression_cost(self.x, self.y, self.theta)

    def train(self, iterations, alpha):
        """Train the linear regression."""
        self.theta, self.cost_history =\
            regression_train(self.x, self.y, self.theta, iterations, alpha)

    def predict(self, values):
        """Make a prediction based on the trained data."""
        if self.normalised:
            values = normalise(values, self.mu, self.sigma)
        values = np.append(1, values)
        return float( values * self.theta )

    def rsquared(self):
        return rsquared(self.y, self.x * self.theta)

def find_baseline():
    features, target = load_boston()

    #find the feature with the biggest R squared
    for i in range(features.shape[1]):
        predictor = features[:, [i]]
        lr = LinearRegression(predictor, target, True)
        lr.train(500, 0.01)
        #lr.plot(filename='plot_%s.png' % i)

def model1():
    features, target = load_boston()

    #choose only feature #13.
    features = features[:,[12]]
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)
    print('theta:')
    print(lr.theta)

    print('R^2:', lr.rsquared())
    lr.plot(filename='baseline.png', xlabel='% of lower status in the population', ylabel='Housing price')

    lr.plot_j()
    plt.savefig('cost_function.png')

def model2():
    features, target = load_boston()

    #choose features #13 & #6.
    features = features[:,[5,12]]
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)

    print('R^2:', lr.rsquared())

def model3():
    features, target = load_boston()

    #choose features #13, #6 & #11.
    features = features[:,[5,10, 12]]
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)

    print('R^2:', lr.rsquared())

def model4():
    features, target = load_boston()

    #including all features.
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)

    print('R^2:', lr.rsquared())

def model5():
    features, target = load_boston()

    #log transformation of DIS
    features[:,7] = np.log(features[:,7])
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)
    print('R^2:', lr.rsquared())

def model6():
    features, target = load_boston()

    #log transformation of DIS, LSTAT &PTRATIO
    features[:, 7] = np.log(features[:, 7])
    features[:, 12] = np.log(features[:, 12])
    features[:, 10] = np.log(features[:, 10])
    lr = LinearRegression(features, target, True)
    lr.train(500, 0.01)
    print('R^2:', lr.rsquared())


def plot():
    data = datasets.load_boston()
    # define the data/predictors as the pre-set feature names
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # Put the target (housing value -- MEDV) in another DataFrame
    target = pd.DataFrame(data.target, columns=["MEDV"])
    plotted_data = df.assign(MEDV=target)

    sns_plot = sns.jointplot(x="CRIM", y="MEDV", data=plotted_data)

    sns_plot.set_axis_labels(xlabel="Crime rate", ylabel="House price")
    sns_plot.savefig("crime.png")

    sns_plot = sns.jointplot(x="ZN", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="% of residential land", ylabel="House price")
    sns_plot.savefig("land.png")

    sns_plot = sns.jointplot(x="INDUS", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="% of non-retail business acres", ylabel="House price")
    sns_plot.savefig("business.png")

    sns_plot = sns.jointplot(x="CHAS", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Charles river", ylabel="House price")
    sns_plot.savefig("river.png")

    sns_plot = sns.jointplot(x="NOX", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Nitric oxides concentration", ylabel="House price")
    sns_plot.savefig("NOX.png")

    sns_plot = sns.jointplot(x="RM", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Number of rooms", ylabel="House price")
    sns_plot.savefig("RM.png")

    sns_plot = sns.jointplot(x="AGE", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="% of buildings built prior to 1940", ylabel="House price")
    sns_plot.savefig("age.png")

    sns_plot = sns.jointplot(x="DIS", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Distance to employment centres", ylabel="House price")
    sns_plot.savefig("distance.png")

    sns_plot = sns.jointplot(x="RAD", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Accessibility to radial highways", ylabel="House price")
    sns_plot.savefig("highway.png")

    sns_plot = sns.jointplot(x="TAX", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Property tax rate", ylabel="House price")
    sns_plot.savefig("tax.png")

    sns_plot = sns.jointplot(x="PTRATIO", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="Pupil-teacher rate", ylabel="House price")
    sns_plot.savefig("teacher rate.png")

    sns_plot = sns.jointplot(x="B", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="% of blacks", ylabel="House price")
    sns_plot.savefig("black.png")

    sns_plot = sns.jointplot(x="LSTAT", y="MEDV", data=plotted_data)
    sns_plot.set_axis_labels(xlabel="% of lower status", ylabel="House price")
    sns_plot.savefig("LSTAT.png")

if __name__ == "__main__":
    find_baseline()
    plot()
    model1()
    model2()
    model3()
    model4()
    model5()
    model6()




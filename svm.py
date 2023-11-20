import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib as plt

gamma = [0.01, 0.1, 0.5, 1, 5, 10, 50]

def loadData(filename):

    data = pd.read_csv(filename, delimiter=',')

    # Using 70% of the data for training
    trainDataSize = int(0.7 * data.shape[0])
    trainData = data.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_train = trainData.iloc[:, -1].values

    # Using 30% of the data for testing
    testData = data.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_test = testData.iloc[:, -1].values

    # Standardizing Y labels
    y_train = np.where(y_train == 2, -1, 1)
    y_test = np.where(y_test == 2, -1, 1)

    
    return X_train, y_train, X_test, y_test

def svmTrain(filename, gama):

    # Loading the data
    X_train, y_train, X_test, y_test = loadData(filename)

    # Defining the decsion variables
    a = cp.Variable((X_train.shape[0], 1))
    b = cp.Variable()
    eta = cp.Variable((X_train.shape[0], 1))

    # Defining the objective function
    objective = cp.Minimize(cp.norm2(a) + gama * cp.norm1(eta))

    # Defining the constraints
    constraints = [cp.multiply(y_train, (a.T @ X_train - b)) >= 1 - eta, eta >= 0]

    # Defining the problem
    problem = cp.Problem(objective, constraints)

    # Solving the problem
    problem.solve()

    return a.value, b.value

def svmTest(filename, gamma):

    # Loading the data
    X_train, y_train, X_test, y_test = loadData(filename)
    a, b = svmTrain(filename, gamma)

    # Calculating the predicted values
    y_pred = np.sign(np.dot(X_test, a) - b)

    # Calculating 0-1 loss
    accuracy = np.sum(y_pred == y_test) / y_test.shape[0]

    return accuracy


if __name__ == '__main__':


    # Training the model
    for gama in gamma:
        a, b = svmTrain('breast-cancer-winsconsin.data', gama)

        # Testing the model
        accuracy = svmTest(a, b, X_test, y_test)

        print('Accuracy for gama = {} is {}'.format(gama, accuracy))
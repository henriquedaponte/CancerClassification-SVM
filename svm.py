import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler



''''
Reason for hadling missing datat using knn imputer:

K-Nearest Neighbors (KNN) Imputation. This method is particularly suitable because:

Data Nature: The dataset is numerical, and KNN works well with such data by finding the 'k' closest samples and imputing missing values based on similarity measures (like Euclidean distance).

Biomedical Relevance: In biomedical datasets, missing values may have patterns that are similar among certain samples. KNN can leverage these patterns effectively.

Flexibility: KNN Imputation is adaptable and can provide more nuanced imputation than simple mean or median replacement.

'''

def loadData(filename):

    data = pd.read_csv(filename, delimiter=',', header=None, na_values='?')

    # Initialize the KNNImputer
    imputer = KNNImputer(n_neighbors=5)  

    # Perform the imputation
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Using 70% of the data for training
    trainDataSize = int(0.7 * data_imputed.shape[0])
    trainData = data_imputed.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_train = trainData.iloc[:, -1].values

    # Using 30% of the data for testing
    testData = data_imputed.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_test = testData.iloc[:, -1].values

    # Standardizing Y labels
    y_train = (np.where(y_train == 2, -1, 1)).reshape(-1, 1)
    y_test = np.where(y_test == 2, -1, 1).reshape(-1, 1)

    # Initialize the scaler
    scaler = StandardScaler()

    # Normalizing the data
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def svmTrain(X_train, y_train, gamma):

    # Defining the decsion variables
    a = cp.Variable((X_train.shape[1], 1))
    b = cp.Variable()
    eta = cp.Variable((X_train.shape[0], 1))

    # Defining the objective function
    objective = cp.Minimize(cp.norm2(a) + gamma * cp.norm1(eta))

    # Defining the constraints
    constraints = [cp.multiply(y_train, (X_train @ a - b)) >= 1 - eta, eta >= 0]

    # Defining the problem
    problem = cp.Problem(objective, constraints)

    # Solving the problem
    problem.solve()

    return a.value, b.value

def svmTest(a, b, X_test, y_test):

    # Calculating the predicted values
    y_pred = np.sign(np.dot(X_test, a) - b)

    # Calculating 0-1 loss
    loss = 1 - np.sum(y_pred == y_test) / y_test.shape[0]

    return loss

def plotData(train_errors, test_errors, gamma):

    train_errors = np.array(train_errors) * 100
    test_errors = np.array(test_errors) * 100

    plt.plot(gamma, train_errors, label='Train Error')
    plt.plot(gamma, test_errors, label='Test Error')
    plt.xlabel('Gamma')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.show()

 


def deployModel(filename):
   
    gamma = [0.01, 0.1, 0.5, 1, 5, 10, 50]

    X_train, y_train, X_test, y_test = loadData(filename)

    train_errors = []
    test_errors = []

    for gama in gamma:
        a, b = svmTrain(X_train, y_train, gama)
        train_error = svmTest(a, b, X_train, y_train)
        test_error = svmTest(a, b, X_test, y_test)

        train_errors.append(train_error)
        test_errors.append(test_error)
         

    plotData(train_errors, test_errors, gamma)

if __name__ == '__main__':
    deployModel('breast-cancer-wisconsin.data')

     
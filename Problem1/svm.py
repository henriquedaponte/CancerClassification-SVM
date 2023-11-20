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
    """
    Load and preprocess the dataset.

    Parameters:
    - filename (str): The path to the dataset file.

    Returns:
    - X_train (np.array): Features for the training data.
    - y_train (np.array): Labels for the training data.
    - X_test (np.array): Features for the test data.
    - y_test (np.array): Labels for the test data.
    """

    # Load the dataset, handling missing values with KNN imputation, and split into training and testing sets.
    data = pd.read_csv(filename, delimiter=',', header=None, na_values='?')

    # Initialize the imputer and apply it to fill in missing values based on neighboring samples.
    imputer = KNNImputer(n_neighbors=5)
    data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Determine the size for training data (80% of the total dataset)
    trainDataSize = int(0.8 * data_imputed.shape[0])

    # Split the imputed data into training and testing sets
    trainData = data_imputed.iloc[:trainDataSize, :]
    X_train = trainData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_train = trainData.iloc[:, -1].values
    testData = data_imputed.iloc[trainDataSize:, :]
    X_test = testData.iloc[:, 1:-1].values # Ignoring the first column (id)
    y_test = testData.iloc[:, -1].values

    # Standardizing Y labels
    y_train = (np.where(y_train == 2, -1, 1)).reshape(-1, 1)
    y_test = np.where(y_test == 2, -1, 1).reshape(-1, 1)

    # Normalize the features to have zero mean and unit variance, as this often improves the performance of SVMs.
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform both training and test data.
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test

def svmTrain(X_train, y_train, gamma):
    """
    Train an SVM classifier using cvxpy.

    Parameters:
    - X_train (np.array): Training features.
    - y_train (np.array): Training labels.
    - gamma (float): The regularization parameter.

    Returns:
    - a (cp.Variable): The learned weight vector.
    - b (cp.Variable): The learned bias term.
    """

    # Define the decision variables for SVM's linear weights and bias term.
    a = cp.Variable((X_train.shape[1], 1))
    b = cp.Variable()

    # Slack variable eta for handling non-linearly separable data
    eta = cp.Variable((X_train.shape[0], 1))

    # Defining the objective function
    # The objective function is composed of the norm of 'a' and the L1 norm of slack variables, weighted by gamma.
    objective = cp.Minimize(cp.norm2(a) + gamma * cp.norm1(eta))

    # Defining the constraints
    constraints = [cp.multiply(y_train, (X_train @ a - b)) >= 1 - eta, eta >= 0]

    # Defining the problem
    problem = cp.Problem(objective, constraints)

    # Solve the optimization problem and return the learned parameters.
    problem.solve()

    return a.value, b.value

def svmTest(a, b, X_test, y_test):
    """
    Test the SVM classifier and calculate the 0-1 loss.

    Parameters:
    - a (cp.Variable): The learned weight vector from svmTrain.
    - b (cp.Variable): The learned bias term from svmTrain.
    - X_test (np.array): Test features.
    - y_test (np.array): Test labels.

    Returns:
    - loss (float): The calculated 0-1 loss.
    """

   # Predict labels for the test set and compare with the true labels to compute the loss.
    y_pred = np.sign(np.dot(X_test, a) - b)

    # Calculating 0-1 loss
    # 0-1 loss is calculated as the fraction of misclassified examples.
    loss = 1 - np.sum(y_pred == y_test) / y_test.shape[0]

    return loss

def plotData(train_errors, test_errors, gamma):
    """
    Plot the training and testing errors against values of gamma.

    Parameters:
    - train_errors (list): List of training errors for each gamma.
    - test_errors (list): List of testing errors for each gamma.
    - gamma (list): List of gamma values used in training.
    """

    # Convert errors to percentages for easier interpretation.
    train_errors = np.array(train_errors) * 100
    test_errors = np.array(test_errors) * 100

    # Plot the training and test errors against the gamma values.
    plt.plot(gamma, train_errors, label='Train Error')
    plt.plot(gamma, test_errors, label='Test Error')
    plt.xlabel('Gamma')
    plt.ylabel('Error (%)')
    plt.legend()
    plt.show()

def deployModel(filename):
    """
    Main function to deploy the SVM model.

    Parameters:
    - filename (str): The path to the dataset file.
    """
   
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

''''
Homework question:

it appears that as gamma increases, the training error decreases slightly while the test 
error remains relatively flat. This indicates that the model may be fitting the training data
more closely as gamma increases.
'''
     
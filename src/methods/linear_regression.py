import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object. 
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self, lmda, task_kind ="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        N, D = training_data.shape
        X = np.hstack([np.ones((N, 1)), training_data])  # bias term with ones its shape (N, D+1)
        
        # Regularization matrix: 
        I = np.eye(D+1)
        
        # Closed form solution with regularization for ridge regression
        self.weights = np.linalg.inv(X.T @ X + self.lmda * I) @ (X.T @ training_labels)

        pred_labels = X @ self.weights
        ###
        ##

        return pred_labels 


    def predict(self, test_data):
        """
            Runs prediction on the test data.
            
            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        N, D = test_data.shape
        X_test = np.hstack([np.ones((N, 1)), test_data])  # bias term with ones its shape (N, D+1)
    
        test_labels = X_test @ self.weights 

        ###
        ##
        return test_labels 

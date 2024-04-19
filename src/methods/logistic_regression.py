import numpy as np

from ..utils import get_n_classes, label_to_onehot, onehot_to_label , append_bias_term


class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500, task_kind ="classification"):
        """
        Initialize the new object (see dummy_methods.py)
        and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind
        


    def fit(self,training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        training_data_scaled = append_bias_term(training_data)
        D = training_data_scaled.shape[1]  # number of features
        C = get_n_classes(training_labels) # number of classes
        label_onehot = label_to_onehot(training_labels,C)
        self.weights = np.random.normal(0, 0.1, (D, C))
        for i in range(self.max_iters):
            logits = training_data_scaled @  self.weights
            exp_logits = np.exp(logits)
            probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            grad = training_data_scaled.T @ (probabilities - label_onehot)
            self.weights = self.weights - self.lr * grad
            

        
        pred_labels = self.predict(training_data)

        
        ###
        ##

        return pred_labels


    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        #
        test_data_scaled=append_bias_term(test_data)
        exp_logits = np.exp(test_data_scaled @ self.weights)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
     # Predict labels
        pred_labels= onehot_to_label(probabilities)
        pred_labels = np.argmax(probabilities, axis=1)
        return pred_labels
    
    

import numpy as np
class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.training_labels = None
        self.training_data = None
        self.k = k
        self.task_kind = task_kind


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        ##
        ###
        #### YOUR CODE HERE!
       
        self.training_data = training_data

        self.training_labels = training_labels

        # Predicting the labels for the training data
        pred_labels = self.predict(training_data)
        ###
        ##
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        ##
        ###
        #### YOUR CODE HERE!
        
        #ERROR if there's no training data
        if self.training_data is None or self.training_labels is None:
            raise ValueError("Model has not been trained.")



        #implementing the test labels with the right shape
        if self.task_kind == "classification":  # classification
            test_labels = np.zeros(len(test_data), dtype=np.int_)
        elif self.task_kind == "regression":  # regression
            shape = self.training_labels.shape[1]
            test_labels = np.zeros((len(test_data), shape))

        for i, sample in enumerate(test_data):
            #calculating the distance as seen in class
            distances = np.sqrt(np.sum((self.training_data - sample) ** 2, axis=1))

            #finding the smallest/nearest k indices and their values
            k_nearest_indices = np.argsort(distances)[:self.k]

            k_nearest_labels = self.training_labels[k_nearest_indices]

            if self.task_kind == "classification":   #classification
                test_labels[i] = np.argmax(np.bincount(k_nearest_labels))

            elif self.task_kind == "regression":  #regressiom
                test_labels[i] = np.mean(k_nearest_labels, axis=0)
            else:
                raise ValueError("Unknown task kind. Supported options: 'classification', 'regression'.")
        ###
        ##
        return test_labels

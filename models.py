import numpy as np
import matplotlib.pyplot as plt

def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))

class RegularizedLogisticRegression(object):
    '''
    Implement regularized logistic regression for binary classification.

    The weight vector w should be learned by minimizing the regularized risk
    log(1 + exp(-y <w, x>)) + \lambda \|w\|_2^2. In other words, the objective
    function is the log loss for binary logistic regression plus Tikhonov
    regularization with a coefficient of \lambda.
    '''
    def __init__(self):
        self.learningRate = 0.00001 # Feel free to play around with this if you'd like, though this value will do
        self.num_epochs = 10000 # Feel free to play around with this if you'd like, though this value will do
        self.batch_size = 15 # Feel free to play around with this if you'd like, though this value will do
        self.weights = None

        #####################################################################
        #                                                                    #
        #    MAKE SURE TO SET THIS TO THE OPTIMAL LAMBDA BEFORE SUBMITTING    #
        #                                                                    #
        #####################################################################

        self.lmbda = 10 # tune this parameter

    def train(self, X, Y):
        '''
        Train the model, using batch stochastic gradient descent
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            None
        '''
        #[TODO]

        # Initializing the weights Array
        num_features = X.shape[1]
        self.weights = np.zeros ((num_features))

        # Initilializing a number representing the number of examples
        num_examples = Y.size

        # Scrubbing the data to make the number of datapoints divisible by the batch size
        remainder = num_examples % self.batch_size
        for i in range (remainder):
            X = np.delete (X, 0, 0)
            Y = np.delete (Y, 0, 0)

        # Finding the number of batches that need to be created
        num_examples = X.shape[0]
        num_batches = int(num_examples/self.batch_size)

        ## Running the SGD algorithm
        for i in range (self.num_epochs):

            # Shuffling the training examples
            random_state = np.random.get_state ()
            np.random.shuffle (X)
            np.random.set_state (random_state)
            np.random.shuffle (Y)

            # Splitting the data into batches
            batched_data = np.vsplit(X, num_batches)

            for i in range (0, num_batches):

                # Initializing array of gradients
                derivatives_array = np.zeros ((self.batch_size))

                # Calculating the inner products
                batch_array = batched_data[i]
                inner_products = np.matmul (batch_array, self.weights)

                # Applying the sigmoid_function
                probabilities = sigmoid_function(inner_products)

                # Setting partial derivatives for the individual weights_derivatives
                for k in range (0, self.batch_size):

                    if 1 == Y[i * self.batch_size + k]:
                        derivatives_array[k] = probabilities[k] - 1
                    else:
                        derivatives_array[k] = probabilities[k]

                # Finding the outer product to be subtracted from the weights Array
                weights_derivatives = np.dot (np.transpose(batch_array), derivatives_array)

                # Completing the regularization
                weights_derivatives = weights_derivatives + (2 * self.lmbda * self.weights)

                # Updating the weights
                self.weights = self.weights - ((self.learningRate * weights_derivatives)/self.batch_size)



    def predict(self, X):
        '''
        Compute predictions based on the learned parameters and examples X
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
        @return:
            A 1D Numpy array with one element for each row in X containing the predicted class.
        '''
        # #[TODO] COMPLETE

        # Creating an array to hold results
        num_examples = X.shape[0]
        result_array = np.zeros ((num_examples))

        # Computing inner products from weight vectors and creating a container for probabilities
        inner_products = np.matmul(X, self.weights)
        probabilities = sigmoid_function (inner_products)
        # Setting the result array
        for i in range (num_examples):
            if probabilities[i] > .5:
                result_array[i] = 1
        return result_array


    def accuracy(self,X, Y):
        '''
        Output the accuracy of the trained model on a given testing dataset X and labels Y.
        @params:
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
        @return:
            a float number indicating accuracy (between 0 and 1)
        '''
        #[TODO] COMPLETE
        predicted_model = self.predict(X)
        num_elements = X.shape[0]
        num_correct = 0

        for i in range (num_elements):
            if predicted_model[i] == Y[i]:
                num_correct += 1

        return (num_correct/num_elements)

    def runTrainTestValSplit(self, lambda_list, X_train, Y_train, X_val, Y_val):
        '''
        Given the training and validation data, fit the model with training data and test it with
        respect to each lambda. Record the training error and validation error.
        @params:
            lambda_list: a list of lambdas
            X_train: a 2D Numpy array for trainig where each row contains an example,
            padded by 1 column for the bias
            Y_train: a 1D Numpy array for training containing the corresponding labels for each example
            X_val: a 2D Numpy array for validation where each row contains an example,
            padded by 1 column for the bias
            Y_val: a 1D Numpy array for validation containing the corresponding labels for each example
        @returns:
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
        '''
        train_errors = []
        val_errors = []

        #[TODO] train model and calculate train and validation errors here for each lambda
        for i in lambda_list:
            self.lmbda = i
            self.train (X_train, Y_train)
            train_errors.append((1 - self.accuracy(X_train, Y_train)))
            val_errors.append((1 - self.accuracy(X_val, Y_val)))

        return train_errors, val_errors

    def _kFoldSplitIndices(self, dataset, k):
        '''
        Helper function for k-fold cross validation. Evenly split the indices of a
        dataset into k groups.

        For example, indices = [0, 1, 2, 3] with k = 2 may have an output
        indices_split = [[1, 3], [2, 0]].

        Please don't change this.
        @params:
            dataset: a Numpy array where each row contains an example
            k: an integer, which is the number of folds
        @return:
            indices_split: a list containing k groups of indices
        '''
        num_data = dataset.shape[0]
        fold_size = int(num_data / k)
        indices = np.arange(num_data)
        np.random.shuffle(indices)
        indices_split = np.split(indices[:fold_size*k], k)
        return indices_split

    def runKFold(self, lambda_list, X, Y, k = 3):
        '''
        Run k-fold cross validation on X and Y with respect to each lambda. Return all k-fold
        errors.

        Each run of k-fold involves k iterations. For an arbitrary iteration i, the i-th fold is
        used as testing data while the rest k-1 folds are training data. The k results are
        averaged as the cross validation error.
        @params:
            lambda_list: a list of lambdas
            X: a 2D Numpy array where each row contains an example, padded by 1 column for the bias
            Y: a 1D Numpy array containing the corresponding labels for each example
            k: an integer, which is the number of folds, k is 3 by default
        @return:
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        '''
        k_fold_errors = []
        for lmbda in lambda_list:
            self.lmbda = lmbda
            #[TODO] call _kFoldSplitIndices to split indices into k groups randomly
            split_indices = self._kFoldSplitIndices (X, k)

            #[TODO] for each iteration i = 1...k, train the model using lmbda
            # on k−1 folds of data. Then test with the i-th fold.
            sum_error = 0

            for i in range (k):
                training_data_array = np.copy(X)
                training_label_array = np.copy(Y)
                test_indices = split_indices[i]

                ## Creating our testing sets
                testing_data_array = np.zeros((X.shape[0]))
                for i in test_indices:
                    testing_data_array = X[test_indices]
                    testing_label_array = Y[test_indices]

                ## Creating our training sets
                training_data_array = np.delete (training_data_array, test_indices, 0)
                training_label_array = np.delete (training_label_array, test_indices, 0)

                ## Training our models
                self.train(training_data_array, training_label_array)

                ## Adding the total error from this batch to our sum_error
                sum_error += (1 - self.accuracy(testing_data_array, testing_label_array))

            #[TODO] calculate and record the cross validation error by averaging total errors
            average_error = sum_error/k
            k_fold_errors.append(average_error)

        return k_fold_errors

    def plotError(self, lambda_list, train_errors, val_errors, k_fold_errors):
        '''
        Produce a plot of the cost function on the training and validation sets, and the
        cost function of k-fold with respect to the regularization parameter lambda. Use this plot
        to determine a valid lambda.
        @params:
            lambda_list: a list of lambdas
            train_errors: a list of training errors with respect to the lambda_list
            val_errors: a list of validation errors with respect to the lambda_list
            k_fold_errors: a list of k-fold errors with respect to the lambda_list
        @return:
            None
        '''
        plt.figure()
        plt.semilogx(lambda_list, train_errors, label = 'training error')
        plt.semilogx(lambda_list, val_errors, label = 'validation error')
        plt.semilogx(lambda_list, k_fold_errors, label = 'k-fold error')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.show()

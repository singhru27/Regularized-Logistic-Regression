import numpy as np
import pandas as pd
from models import RegularizedLogisticRegression

def extract():
    X_train = pd.read_csv('./data/X_train.csv',header=None)
    Y_train = pd.read_csv('./data/y_train.csv',header=None)
    X_val = pd.read_csv('./data/X_val.csv',header=None)
    Y_val = pd.read_csv('./data/y_val.csv',header=None)

    Y_train = np.array([i[0] for i in Y_train.values])
    Y_val = np.array([i[0] for i in Y_val.values])

    X_train = np.append(X_train, np.ones((len(X_train), 1)), axis=1)
    X_val = np.append(X_val, np.ones((len(X_val), 1)), axis=1)

    return X_train, X_val, Y_train, Y_val

def main():
    X_train, X_val, Y_train, Y_val = extract()
    X_train_val = np.concatenate((X_train, X_val))
    Y_train_val = np.concatenate((Y_train, Y_val))

    RR = RegularizedLogisticRegression()
    RR.train(X_train, Y_train)
    print('Train Accuracy: ' + str(RR.accuracy(X_train, Y_train)))
    print('Validation Accuracy: ' + str(RR.accuracy(X_val, Y_val)))

    #[TODO] Once implemented, uncomment the following lines of code and:
    # 1. implement runTrainTestValSplit to get the training and validation errors of our 70-15-15
    #    split to the original dataset
    # 2. implement runKFold to generate errors of each lambda, where k = 3 in this assignment
    # 3. call plotError to plot those errors with respect to lambdas

    # lambda_list = [1000, 100, 10, 1, 0.1, 0.01, 0.001]
    # train_errors, val_errors = RR.runTrainTestValSplit(lambda_list, X_train, Y_train, X_val, Y_val)
    # k_fold_errors = RR.runKFold(lambda_list, X_train_val, Y_train_val, 3)
    # print(lambda_list)
    # print(train_errors, val_errors, k_fold_errors)
    # RR.plotError(lambda_list, train_errors, val_errors, k_fold_errors)

if __name__ == '__main__':
    main()

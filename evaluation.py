import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def corr_spearman(y_true, y_pred):
    """
    Calculates the Spearman rank correlation between y_true and y_pred.

    Args:
    y_true (array-like): The true target values.
    y_pred (array-like): The predicted target values.

    Returns:
    The Spearman rank correlation coefficient between y_true and y_pred.
    """
    return spearmanr(y_true, y_pred).correlation


def eval_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics for a binary classification problem.

    Args:
    y_true (array-like): The true target values.
    y_pred (array-like): The predicted target values.

    Returns:
    A list containing the accuracy, precision, recall, and F1 score.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    scores = [acc, prec, rec, f1]
    return scores



# This function is used to evaluate the model by taking the mean of the accuracy over number_of_state times of split between train and test 
def evaluate_model(X, y, model, stratify, test_size=0.33, acc=corr_spearman, number_of_states=100):
    """
    Evaluates the performance of a given model on a dataset.

    Args:
    X (array-like): Input data.
    y (array-like): Target data.
    model (object): An instance of the model to be trained and evaluated.
    stratify (array-like): Specifies the stratification split for y.
    test_size (float): Proportion of the dataset to be included in the test split.
    acc (function): A function that calculates a score between y_true and y_pred.
    number_of_states (int): The number of random states to use in train-test splitting.

    Returns:
    A numpy array containing the mean train and test scores across all random states.
    """

    score = []
    for i in range(number_of_states):
        # split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i, stratify=stratify)

        # convert to numpy arrays
        X_train = np.array(X_train)
        X_test = np.array(X_test)

        # train the model
        model.fit(X_train, y_train)

        # calculate the training score
        train_score = acc(y_train, model.predict(X_train))

        # calculate the test score
        y_predict = model.predict(X_test)
        test_score = acc(y_test, y_predict.reshape(-1))

        # append the scores to the list of scores
        score.append([train_score, test_score])

    # calculate the mean train and test scores across all random states
    return np.mean(np.array(score), axis=0)

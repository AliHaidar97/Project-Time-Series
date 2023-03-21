import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def corr_spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation



def eval_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro')
    rec = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    scores = [acc, prec, rec, f1]
    return scores


# This function is used to evaluate the model by taking the mean of the accuracy over number_of_state times of split between train and test 
def evaluate_model(X, y, model, stratify ,  test_size=0.33, acc = corr_spearman, number_of_states = 100):
    
    score = []
    for i in range(number_of_states):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i, stratify=stratify)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
        model.fit(X_train,y_train)
        
        train_score = acc(y_train, model.predict(X_train))
        
        y_predict = model.predict(X_test)
        test_score = acc(y_test , y_predict.reshape(-1))
        
        score.append([train_score,test_score])
 
    return np.mean(np.array(score),axis=0)
import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

def corr_spearman(y_true, y_pred):
    return spearmanr(y_true, y_pred).correlation


class CustomLinearModel(BaseEstimator):
    """
    Linear model: Y = XB, fit by minimizing the provided loss_function
    with L2 regularization or L1 regularization
    """
    def __init__(self, X=None, Y=None, sample_weights=None, beta_init=None, 
                 regularization=0.01, alpha = 1.2, optim='BFGS', l = 'l1'):
        self.regularization = regularization
        self.beta = None
        self.sample_weights = sample_weights
        self.beta_init = beta_init
        self.alpha = alpha
        self.X = X
        self.Y = Y
        assert(l  in ['l1','l2'],"Your big brother says : keep it safe and only use l1 or l2 :)")
        self.l = l
        self.optim = optim
        self.coef_ = self.beta
    
       
    def clip_beta(self):
        # This function helps to remove the feature that are noisy
        self.beta[np.abs(self.beta) < 1e-4] = 0
    
    def score(self,X,y_true):
        y_pred = self.predict(X)
        return corr_spearman(y_pred,y_true)
        
    
    def predict(self, X):
        prediction = X@self.beta
        return(prediction)

    def model_error(self):
        y_pred = self.predict(self.X)
        error = np.mean(np.abs(y_pred-self.Y)**(self.alpha))
        return(error)
    
    def l2_regularized_loss(self, beta):
        self.beta = beta
        return(self.model_error() + \
               sum(self.regularization*np.abs(np.array(self.beta))**2))
    
    def l1_regularized_loss(self, beta):
        self.beta = beta
        return(self.model_error() + \
               sum(self.regularization*np.abs(np.array(self.beta))))
    
    
    def fit(self, X, Y, maxiter=10000):       
        self.X = np.array(X)
        self.Y = np.array(Y) 
        # Initialize beta estimates (you may need to normalize
        # your data and choose smarter initialization values
        # depending on the shape of your loss function)
        if type(self.beta_init)==type(None):
            # set beta_init = 1 for every feature
            self.beta_init = np.array([1]*self.X.shape[1])
        else: 
            self.beta_init = np.array([1]*self.X.shape[1])
            self.beta = None
            
        if self.beta!=None and all(self.beta_init == self.beta):
            print("Model already fit once; continuing fit with more itrations.")
        
        if(self.l == 'l2'):
            res = minimize(self.l2_regularized_loss, self.beta_init, 
                        method=self.optim, options={'maxiter': maxiter,'disp':False})
        else:
            res = minimize(self.l1_regularized_loss, self.beta_init, 
                        method=self.optim, options={'maxiter': maxiter,'disp':False})
            
        self.beta = res.x
        self.beta_init = self.beta
        self.clip_beta()
        self.coef_ = self.beta


# This function is used to evaluate the model by taking the mean of the accuracy over number_of_state times of split between train and test 
def evaluate_model(X, y, model, stratify ,  test_size=0.33, acc = corr_spearman, number_of_states = 100):
    
    score = []
    for i in range(number_of_states):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=i, stratify=stratify)
        
        X_train = X_train.drop(['COUNTRY_split'],axis=1)
        X_test = X_test.drop(['COUNTRY_split'],axis=1)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
        model.fit(X_train,y_train)
        
        train_score = acc(y_train, model.predict(X_train))
        
        y_predict = model.predict(X_test)
        test_score = acc(y_test , y_predict.reshape(-1))
        
        score.append([train_score,test_score])
        
    return np.mean(np.array(score),axis=0)

# This function is used to evaluate the model by taking the mean of the accuracy over number_of_state times of split between train and test for each country
def evaluate_model_by_country(X, y, model, stratify ,  test_size=0.33, acc = corr_spearman, number_of_states = 100):
    
    score = []
    for i in range(number_of_states):
        
        # We train on a balanced training set and we test on a data set that contains only FR or only DE.
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,random_state=i, stratify=stratify)
         
        # just test on FR 
        mask = (X_test['COUNTRY_split']==-1)
        X_test_FR = X_test[mask].copy()
        X_test_FR = X_test_FR.drop(['COUNTRY_split'],axis=1)
        y_train = np.array(y_train)
        y_test_FR = np.array(y_test)[np.array(mask)].copy()
        
        # just test on DE
        mask = (X_test['COUNTRY_split']==1)
        X_test_DE = X_test[mask].copy()
        X_test_DE = X_test_DE.drop(['COUNTRY_split'],axis=1)
        y_test_DE = np.array(y_test)[np.array(mask)].copy()
      
        X_train = X_train.drop(['COUNTRY_split'],axis=1)
        X_test = X_test.drop(['COUNTRY_split'],axis=1)
        X_train = np.array(X_train)
        X_test = np.array(X_test)
      
        X_test_FR = np.array(X_test_FR)
        X_test_DE = np.array(X_test_DE)
      
        model.fit(X_train,y_train)
        train_score = acc(y_train, model.predict(X_train))
        
        y_predict = model.predict(X_test)
        test_score = acc(y_test , y_predict.reshape(-1))
        
        y_predict_FR = model.predict(X_test_FR)
        test_score_FR = acc(np.array(y_test_FR) , y_predict_FR.reshape(-1))
        
        y_predict_DE = model.predict(X_test_DE)
        test_score_DE = acc(np.array(y_test_DE) , y_predict_DE.reshape(-1))
      
        score.append([train_score, test_score, test_score_FR, test_score_DE])
        
    return np.mean(np.array(score),axis=0)
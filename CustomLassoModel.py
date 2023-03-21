import numpy as np 
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
import evaluation


class CustomLassoModel(BaseEstimator):
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
        return evaluation.corr_spearman(y_pred,y_true)
        
    
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


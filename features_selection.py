import pandas as pd 
import numpy as np 
import model 
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from sklearn.linear_model import LinearRegression
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import MIFS

def evaluate_model_features(X_train , y_train, list_features, list_scores, number_of_states = 5, test_size = 0.3, name = '-'):
    #Plot a graph showing the score
    plt.figure(figsize=(20, 7))
    plt.barh(list_features, list_scores, color='r')
    plt.title(name + " Score of each feature")
    plt.show();

    cols = list_features

    train_score = []
    test_score = []
    test_score_FR = []
    test_score_DE = []
    number_features = []
    features_selected = []

    for i in tqdm(range(2,len(cols))):
        select_features = cols[:i+1] 
        score = model.evaluate_model_by_country(X_train[select_features+['COUNTRY_split']], y_train, LinearRegression(), stratify= X_train['COUNTRY_split'] , number_of_states = number_of_states)
        train_score.append(score[0])     
        test_score.append(score[1])
        test_score_FR.append(score[2])
        test_score_DE.append(score[3])
        number_features.append(len(select_features))
        features_selected.append(select_features)
        
    #plot the accuracy vs the number of features
    plt.figure(figsize=(20, 7))
    plt.plot(np.arange(len(train_score))+1, train_score, label = 'train_score')
    plt.plot(np.arange(len(train_score))+1, test_score, label = 'test_score')
    plt.plot(np.arange(len(train_score))+1, test_score_FR, label = 'test_score_FR')
    plt.plot(np.arange(len(train_score))+1, test_score_DE, label = 'test_score_DE')
    plt.xlabel('number of features')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.plot()
     
    best_score_idx = np.argmax(test_score)
    best_score = test_score[best_score_idx]
    best_score_FR = test_score_FR[best_score_idx]
    best_score_DE = test_score_DE[best_score_idx]
    best_features = features_selected[best_score_idx]  

    return best_score, best_score_FR, best_score_DE, best_features


def lasso_feature_selection(X_train, y_train, number_of_states = 5, test_size = 0.3):
    lasso_parameters = {
    'regularization': np.arange(0, 0.1, 0.005)
    }

    #for alpha in lasso_parameters['alpha']:
    alpha = 1.2

    train_score = []
    test_score = []
    test_score_FR = []
    test_score_DE = []
    number_features = []
    features_selected = []

    for regularization in tqdm(lasso_parameters['regularization']):
        l1_lasso_model = model.CustomLinearModel(regularization=regularization, alpha= 1.2, optim = 'BFGS')
        score = model.evaluate_model_by_country(X_train, y_train, l1_lasso_model, stratify = X_train['COUNTRY_split'],  state = 5, test_size=0.3)
        features = list(zip(X_train.columns[:-1],l1_lasso_model.coef_))
        #features to keep
        keep = []
        for i in features:
            if(i[1]!=0):
                keep.append(i[0])
        
        train_score.append(score[0])     
        test_score.append(score[1])
        test_score_FR.append(score[2])
        test_score_DE.append(score[3])
        number_features.append(len(keep))
        features_selected.append(keep)
        
    #plot the accuracy vs the regularization
    plt.figure(figsize=(20, 7))
    plt.subplot(1,2,1)
    plt.plot(lasso_parameters['regularization'], train_score, label = 'train_score')
    plt.plot(lasso_parameters['regularization'], test_score, label = 'test_score')
    plt.plot(lasso_parameters['regularization'], test_score_FR, label = 'test_score_FR')
    plt.plot(lasso_parameters['regularization'], test_score_DE, label = 'test_score_DE')
    plt.xlabel('regularization')
    plt.ylabel('Spearman Correlation')
    plt.legend()
    plt.plot()

    #plot the number of selected features vs regularization
    plt.subplot(1,2,2)
    plt.plot(lasso_parameters['regularization'], number_features, label = 'number of features')
    plt.xlabel('regularization')
    plt.ylabel('number of selected features')
    plt.legend()
    plt.plot()  
    
    best_score_idx = np.argmax(test_score)
    regularization = lasso_parameters['regularization'][best_score_idx]
    best_score = test_score[best_score_idx]
    best_score_FR = test_score_FR[best_score_idx]
    best_score_DE = test_score_DE[best_score_idx]
    best_features = features_selected[best_score_idx]  
    
    
    #Show an histogram of the frequency of selected feature

    freq = {} # Dictionary to calculate the frequency of each feature
    for i in features_selected:
        for j in i:
            if(j not in freq):
                freq[j] = 0
            freq[j] += 1
    freq = dict(sorted(freq.items(), key=lambda x: x[1]))

    plt.figure(figsize=(20, 7))
    plt.barh(list(freq.keys()), freq.values(), color='r')
    plt.barh(best_features, np.ones(len(best_features)), color='g',label='best_features')
    plt.title('Frequency of the features vs the selected features')
    plt.legend()
    plt.plot()
    
    return regularization, best_score, best_score_FR, best_score_DE, best_features


def laplace_features_selection(X_train, y_train, X_test, number_of_states = 5, test_size = 0.3, max_features = 32):
    
   
    kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':1}

    X = pd.concat([X_train.drop(['COUNTRY_split'],axis=1),X_test])

    W = construct_W.construct_W(np.array(X), **kwargs_W)

    score = dict(zip(X.columns,lap_score.lap_score(np.array(X), W=W)))
    score = dict(sorted(score.items(), key=lambda x: x[1]))

    max_features = min(max_features,X.shape[1])
    list_features = list(score.keys())[:max_features]
    list_scores = list(score.values())[:max_features]

    
    return evaluate_model_features(X_train , y_train, list_features, list_scores, number_of_states, test_size,  name = 'Laplace')



def MRMR_features_selection(X_train, y_train, X_test, number_of_states = 5, test_size = 0.3, max_features = 32):
    
   
    X = X_train.drop(['COUNTRY_split'],axis=1)
    max_features = min(max_features,X.shape[1])
    idx, J, M =  MRMR.mrmr(np.array(X),np.array(y_train),n_selected_features= max_features)

    list_features = list(X.columns[idx])
    list_scores = J

    
    return evaluate_model_features(X_train , y_train, list_features, list_scores, number_of_states, test_size, name = 'MRMR')


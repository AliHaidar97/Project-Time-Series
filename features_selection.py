import pandas as pd 
import numpy as np 
import CustomLassoModel 
import utils
import matplotlib.pyplot as plt
from tqdm import tqdm
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
from sklearn.linear_model import LinearRegression
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import DISR
from skfeature.function.information_theoretical_based import MIFS
from skfeature.function.similarity_based import fisher_score
from skfeature.function.information_theoretical_based import MRMR
from skfeature.function.information_theoretical_based import CIFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import make_scorer
import evaluation

def evaluate_model_features(X_train , y_train, model, stratify,  list_features, list_scores, number_of_states = 5, test_size = 0.3, name = '-', plot = True, metric = evaluation.corr_spearman):
    

    cols = list_features

    train_score = []
    test_score = []

    number_features = []
    features_selected = []
    pbar = range(2,len(cols))
    if(plot):
        pbar = tqdm(range(2,len(cols)))
    for i in pbar:
        select_features = cols[:i+1] 
        score = evaluation.evaluate_model(X_train[select_features], y_train, model, stratify= stratify , number_of_states = number_of_states, acc = metric)
        train_score.append(score[0])     
        test_score.append(score[1])
        number_features.append(len(select_features))
        features_selected.append(select_features)
    
    train_score = np.array(train_score)
    test_score = np.array(test_score)
    if(plot == True):
        #plot the accuracy vs the number of features
        plt.figure(figsize=(20, 7))
 
        if(train_score.ndim > 1):
            plt.plot(np.arange(len(train_score))+1, train_score[:,0], label = 'train_score')
            plt.plot(np.arange(len(train_score))+1, test_score[:,0], label = 'test_score')
        else:
            plt.plot(np.arange(len(train_score))+1, train_score, label = 'train_score')
            plt.plot(np.arange(len(train_score))+1, test_score, label = 'test_score')
            
        plt.xlabel('number of features')
        plt.ylabel('Spearman Correlation')
        plt.legend()
        plt.plot()
        plt.show();
    
    if(test_score.ndim > 1 ):
        best_score_idx = np.argmax(test_score[:,0])
    else:
        best_score_idx = np.argmax(test_score)
        
    best_score = test_score[best_score_idx]
   
    best_features = features_selected[best_score_idx]  

    return best_score, best_features


# Sparse Learning based Methods #
# ----------------------------- #

def lasso_feature_selection(X_train, y_train, model, stratify, number_of_states = 5, test_size = 0.3, max_features = 32):
    
    lasso_parameters = {
    'regularization': np.arange(0, 0.1, 0.005)
    }
    
    train_score = []
    test_score = []
    number_features = []
    features_selected = []

    for regularization in tqdm(lasso_parameters['regularization']):
        l1_lasso_model = model(regularization)
        score = evaluation.evaluate_model(X_train, y_train, l1_lasso_model, stratify = stratify,  number_of_states = number_of_states, test_size=test_size)
        features = list(zip(X_train.columns[:-1],l1_lasso_model.coef_))
        #features to keep
        keep = []
        for i in features:
            if(i[1]!=0):
                keep.append(i[0])
        
        train_score.append(score[0])     
        test_score.append(score[1])
        number_features.append(len(keep))
        features_selected.append(keep)
        
    #plot the accuracy vs the regularization
    plt.figure(figsize=(20, 7))
    plt.subplot(1,2,1)
    plt.plot(lasso_parameters['regularization'], train_score, label = 'train_score')
    plt.plot(lasso_parameters['regularization'], test_score, label = 'test_score')

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
    best_features = features_selected[best_score_idx]  
    
    
    #Show an histogram of the frequency of selected feature

    freq = {} # Dictionary to calculate the frequency of each feature
    for i in features_selected:
        for j in i:
            if(j not in freq):
                freq[j] = 0
            freq[j] += 1
    freq = dict(sorted(freq.items(), key=lambda x: x[1]))

    if( X_train.shape[1] < max_features):
        plt.figure(figsize=(20, 7))
        plt.barh(list(freq.keys()), freq.values(), color='r')
        plt.barh(best_features, np.ones(len(best_features)), color='g',label='best_features')
        plt.title('Frequency of the features vs the selected features')
        plt.legend()
        plt.plot()
    
    return regularization, best_score, best_features


# Similarity based Methods #
# ------------------------ #

def laplace_features_selection(X_train, y_train, X_test, models, stratify, var = 1, number_of_states = 5, test_size = 0.3, max_features = 32, plot = True, metric = evaluation.corr_spearman):
    
   
    kwargs_W = {"metric":"euclidean","neighbor_mode":"knn","weight_mode":"heat_kernel","k":5,'t':var}

    X = pd.concat([X_train,X_test])

    W = construct_W.construct_W(np.array(X), **kwargs_W)

    score = dict(zip(X.columns,lap_score.lap_score(np.array(X), W=W)))
    score = dict(sorted(score.items(), key=lambda x: x[1]))

    max_features = min(max_features, X.shape[1])
    list_features = list(score.keys())[:max_features]
    list_scores = list(score.values())[:max_features]
    
    if(plot == True):
        #Plot a graph showing the score
        plt.figure(figsize=(20, 7))
        plt.barh(list_features, list_scores, color='r')
        plt.title('Laplace' + " Score of each feature")
        plt.show();
    
    
    best_score = []
    best_features = []
    

    for model in models:
        if(plot):
            print(type(model).__name__)
        b_score, b_features = evaluate_model_features(X_train , y_train, model, stratify,  list_features, list_scores, number_of_states, test_size,  name = 'Laplace', plot = plot, metric = metric)
        best_score.append(b_score)
        best_features.append(b_features)
    return best_score, best_features


def tune_laplace_features_selection(X_train, y_train, X_test, models, stratify, var = [1], number_of_states = 5, test_size = 0.3, max_features = 32, metric = evaluation.corr_spearman):
    
    scores = []
    for v in tqdm(var):
        score, features = laplace_features_selection(X_train, y_train, X_test, models, stratify, var = v, number_of_states = number_of_states, test_size = test_size, max_features = max_features, plot = False, metric = metric)
        scores.append(score)
    
    idx = np.argmax(scores)
    plt.figure(figsize=(20, 7))
    plt.plot(var, scores, color='r')
    plt.title("Laplace Accuracy for each t")
    plt.show();
    
    return var[idx]


def fisher_feature_selection(X_train, y_train, X_test, models, stratify, number_of_states = 5, test_size = 0.3, max_features = 32,  metric = evaluation.corr_spearman):
    
    X = pd.concat([X_train,X_test])
    max_features = min(max_features, X.shape[1])
    
    list_scores =  fisher_score.fisher_score(np.array(X_train),np.array(y_train))
    idx = fisher_score.feature_ranking(list_scores)
    list_features = list(X_train.columns[idx])

    
    #Plot a graph showing the score
    plt.figure(figsize=(20, 7))
    plt.barh(list_features, list_scores, color='r')
    plt.title('Fisher' + " Score of each feature")
    plt.show();
    
    
    best_score = []
    best_features = []
    

    for model in models:
        print(type(model).__name__)
        b_score, b_features = evaluate_model_features(X_train , y_train, model, stratify,  list_features, list_scores, number_of_states, test_size,  name = 'Fisher', metric = metric)
        best_score.append(b_score)
        best_features.append(b_features)
    return best_score, best_features
    
#  Information Theoretical based Methods #
# -------------------------------------- #

def MRMR_features_selection(X_train, y_train, X_test, models, stratify, number_of_states = 5, test_size = 0.3, max_features = 32,  metric = evaluation.corr_spearman):
    
   
    X = X_train.copy()
    max_features = min(max_features, X.shape[1])
    idx, J, M =  MRMR.mrmr(np.array(X), np.array(y_train), n_selected_features= max_features)
    
    order = np.argsort(J)[::-1]
    idx = idx[order]
    J = J[order]
    M = M[order]
    
    list_features = list(X.columns[idx])
    list_scores = J

    plt.figure(figsize=(20, 7))
    plt.barh(list_features, M, color='r')
    plt.title("The Mutual Information" + " score of each feature")
    plt.show();
    
    #Plot a graph showing the score
    plt.figure(figsize=(20, 7))
    plt.barh(list_features, list_scores, color='r')
    plt.title('MRMR' + " Score of each feature")
    plt.show();
    
    
    best_score = [] 
    best_features = []

    for model in models:
        print(type(model).__name__)
        b_score,b_features = evaluate_model_features(X_train , y_train, model, stratify, list_features, list_scores, number_of_states, test_size, name = 'MRMR', metric = metric)
        best_score.append(b_score)
        best_features.append(b_features)
    
    return best_score, best_features



def CIFE_features_selection(X_train, y_train, X_test, models, stratify, number_of_states = 5, test_size = 0.3, max_features = 32,  metric = evaluation.corr_spearman):
    
    
    X = X_train.copy()
    max_features = min(max_features, X.shape[1])
    idx, J, M =  CIFE.cife(np.array(X_train),np.array(y_train),n_selected_features= max_features)
    
    order = np.argsort(J)[::-1]
    idx = idx[order]
    J = J[order]
    M = M[order]
    
    list_features = list(X.columns[idx])
    list_scores = J

    plt.figure(figsize=(20, 7))
    plt.barh(list_features, M, color='r')
    plt.title("The Mutual Information" + " score of each feature")
    plt.show();
    
    #Plot a graph showing the score
    plt.figure(figsize=(20, 7))
    plt.barh(list_features, list_scores, color='r')
    plt.title('CIFE' + " Score of each feature")
    plt.show();
    
    
    best_score = [] 
    best_features = []

    for model in models:
        print(type(model).__name__)
        b_score,b_features = evaluate_model_features(X_train , y_train, model, stratify, list_features, list_scores, number_of_states, test_size, name = 'CIFE', metric = metric)
        best_score.append(b_score)
        best_features.append(b_features)
    
    return best_score, best_features

#  Sequential  Methods #
# -------------------------------------- #


def sequential_features_selection(X_train, y_train, models, stratify, number_of_states = 5, test_size = 0.3, max_features = 32, metric = evaluation.corr_spearman):

    max_features = max(max_features, X_train.shape[1])
    
    best_score = [] 
    best_features = []

    for model in models:
        print(type(model).__name__)
        test_score = []
        number_features = []
        features_selected = []
        
        for m in tqdm(range(2,max_features)):
            sfs = SequentialFeatureSelector(model, n_features_to_select=m, direction='backward', scoring = make_scorer(metric))
            sfs.fit(X_train, np.array(y_train))
            features = X_train.columns[sfs.get_support()]
            features_selected.append(features)
            score = evaluation.evaluate_model(X_train[features], y_train, model, stratify = stratify,  number_of_states = number_of_states, test_size=test_size, acc = metric)
            test_score.append(score[1])
            number_features.append(m)
        
        #plot the accuracy vs the regularization
        plt.figure(figsize=(20, 7))
        plt.subplot(1,2,1)
        plt.plot(number_features, test_score, label = 'test_score')

        plt.xlabel('number of features')
        plt.ylabel('Spearman Correlation')
        plt.legend()
        plt.plot()

        best_score_idx = np.argmax(test_score)
        best_score.append(test_score[best_score_idx])
        best_features.append(features_selected[best_score_idx]) 
        
        
    return best_score, best_features 
        
        
def greedy_features_selection(X_train, y_train, models, stratify, threshold = 0.005, number_of_states = 5, test_size = 0.3, max_features = 32,  metric = evaluation.corr_spearman):

    max_features = max(max_features, X_train.shape[1])
    
    best_score = [] 
    best_features = []

    for model in models:
        print(type(model).__name__)
        test_score = []
        number_features = []
        features_selected = []
        
        best_curr = 0
        
        for m in tqdm(range(0,max_features)):
            features = features_selected + [X_train.columns[m]]
            score = evaluation.evaluate_model(X_train[features], y_train, model, stratify = stratify,  number_of_states = number_of_states, test_size=test_size, acc = metric)[1]
            
            if(isinstance(score, np.ndarray)):
                if((score[0] - best_curr) >= threshold):
                    features_selected = features
                    test_score.append(score)
                    best_curr = score[0]
                    number_features.append(len(features_selected))
            else:
                if((score - best_curr) >= threshold):
                    features_selected = features
                    test_score.append(score)
                    best_curr = score
                    number_features.append(len(features_selected))
                    
        
        #plot the accuracy vs the regularization
        plt.figure(figsize=(20, 7))
        plt.subplot(1,2,1)
        plt.plot(number_features, test_score, label = 'test_score')

        plt.xlabel('number of features')
        plt.ylabel('Spearman Correlation')
        plt.legend()
        plt.plot()
        test_score = np.array(test_score)
        if(test_score.ndim == 2):
            best_score_idx = np.argmax(test_score[:,0])
        else:
            best_score_idx = np.argmax(test_score)
        best_score.append(test_score[best_score_idx])
        best_features.append(features_selected) 
        
        
    return best_score, best_features 
from sklearn.impute import KNNImputer
import numpy as np 
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

def regress_var(df, x_columns, y_column, model, out = True):
    temp = df.dropna().copy()
    X = np.array(temp[x_columns])
    y = np.array(temp[y_column])
    model.fit(X,y)
    if(out):
        print("Model_regression " + y_column + " score :", model.score(X,y))
    return df.apply(lambda row: model.predict(np.array(row[x_columns]).reshape(-1,len(x_columns)))[0] if(np.isnan(row[y_column])) else row[y_column], axis=1)


def clean_knn(all_data_clean, k =5):
    imputer = KNNImputer(n_neighbors=k)
    remove_columns = ['ID','DAY_ID','TARGET','train']
    keep = all_data_clean.columns.difference(remove_columns)
    all_data_clean[keep] = imputer.fit_transform(all_data_clean[keep])
    return all_data_clean
    
def clean_regression(all_data_clean):
    
  
    all_data_clean['DE_FR_EXCHANGE'] = all_data_clean['DE_FR_EXCHANGE'].fillna(all_data_clean['DE_FR_EXCHANGE'].mean(numeric_only=True))
    all_data_clean['FR_DE_EXCHANGE'] = all_data_clean['FR_DE_EXCHANGE'].fillna(all_data_clean['FR_DE_EXCHANGE'].mean(numeric_only=True))
    
    x_columns = ['DE_FR_EXCHANGE']
    y_column = 'DE_NET_EXPORT'
    all_data_clean[y_column] = regress_var(all_data_clean, x_columns, y_column, LinearRegression(), out=True)
    all_data_clean['DE_NET_IMPORT'] = - all_data_clean['DE_NET_EXPORT']
    
    x_columns = ['FR_DE_EXCHANGE']
    y_column = 'FR_NET_EXPORT'
    all_data_clean[y_column] = regress_var(all_data_clean, x_columns, y_column, LinearRegression(), out=True)
    all_data_clean['FR_NET_IMPORT'] = - all_data_clean['FR_NET_EXPORT']
    
    all_data_clean = all_data_clean.fillna(all_data_clean.mean(numeric_only=True))
    
    return all_data_clean

def remove_features(all_data_clean):
    
    all_data_clean = all_data_clean.drop(['DE_FR_EXCHANGE', 'FR_NET_IMPORT','DE_NET_IMPORT'],axis=1)
    all_data_clean['FR_NET_EXPORT'] -= all_data_clean['FR_DE_EXCHANGE']
    all_data_clean['DE_NET_EXPORT'] += all_data_clean['FR_DE_EXCHANGE']
    
    
    all_data_clean['DE_CONSUMPTION_RENEWABLE'] = all_data_clean['DE_CONSUMPTION'] - all_data_clean['DE_RESIDUAL_LOAD']
    all_data_clean['FR_CONSUMPTION_RENEWABLE'] = all_data_clean['FR_CONSUMPTION'] - all_data_clean['FR_RESIDUAL_LOAD']
    
    return all_data_clean

def add_clusters(k, all_data_clean,cols,c):
    
    X_season = all_data_clean[cols]
    #all_data_clean = all_data_clean.drop(cols,axis=1)
    kmeans = KMeans(n_clusters=k,random_state = 0).fit(np.array(X_season))
    all_data_clean[c] = kmeans.predict(np.array(X_season))
    all_data_clean = pd.get_dummies(all_data_clean, columns=[c])
    return all_data_clean


def replace_outliers(all_data_clean, cols, threshold = 3):
    for c in cols :
        upper_limit = all_data_clean[c].mean() + threshold*all_data_clean[c].std()
        lower_limit = all_data_clean[c].mean() - threshold*all_data_clean[c].std()    
        all_data_clean[c] = np.where(
            (all_data_clean[c]>upper_limit) & (all_data_clean['train']==0),
            upper_limit,
            np.where(
                (all_data_clean[c]<lower_limit) & (all_data_clean['train']==0),
                lower_limit,
                all_data_clean[c]
            ))
    return all_data_clean

def remove_outliers(all_data_clean, cols, threshold = 3):
    for c in cols :
        upper_limit = all_data_clean[c].mean() + threshold*all_data_clean[c].std()
        lower_limit = all_data_clean[c].mean() - threshold*all_data_clean[c].std()    
        all_data_clean[c] = np.where(
            (all_data_clean[c]>upper_limit) & (all_data_clean['train']==1),
            np.nan,
            np.where(
                (all_data_clean[c]<lower_limit) & (all_data_clean['train']==1),
                np.nan,
                all_data_clean[c]
            ))
    all_data_clean = all_data_clean.dropna()
    return all_data_clean


def add_new_features(all_data_clean):
    
    all_data_clean['DE_FLOW_GAS'] = all_data_clean['DE_GAS']*all_data_clean['GAS_RET']
    all_data_clean['DE_FLOW_COAL'] = all_data_clean['DE_COAL']*all_data_clean['COAL_RET']
    all_data_clean['DE_FLOW_LIGNITE'] = all_data_clean['DE_LIGNITE']*all_data_clean['CARBON_RET']


    all_data_clean['FR_FLOW_GAS'] = all_data_clean['FR_GAS']*all_data_clean['GAS_RET']
    all_data_clean['FR_FLOW_COAL'] = all_data_clean['FR_COAL']*all_data_clean['COAL_RET']

    #cols = ['DE_RAIN','FR_RAIN','DE_WIND','FR_WIND','DE_TEMP','FR_TEMP']
    #all_data_clean = add_clusters(4,all_data_clean,cols,'season')
    
    
    #all_data_clean["FR_PRODUCTION"] = all_data_clean['FR_GAS'] + all_data_clean['FR_COAL'] + all_data_clean['FR_HYDRO'] + all_data_clean['FR_NUCLEAR'] + all_data_clean['FR_SOLAR'] + all_data_clean['FR_WINDPOW']
    #all_data_clean["DE_PRODUCTION"] = all_data_clean['DE_GAS'] + all_data_clean['DE_COAL'] + all_data_clean['DE_HYDRO'] + all_data_clean['DE_NUCLEAR'] + all_data_clean['DE_SOLAR'] + all_data_clean['DE_WINDPOW'] + all_data_clean['DE_LIGNITE']

    #all_data_clean['FR_NEED'] = all_data_clean["FR_CONSUMPTION"] - all_data_clean['FR_PRODUCTION']  
    #all_data_clean['DE_NEED'] = all_data_clean["DE_CONSUMPTION"] - all_data_clean['DE_PRODUCTION'] 

    #all_data_clean['FR_NEED_RATIO'] =   all_data_clean['FR_NEED'] / all_data_clean["FR_PRODUCTION"]
    #all_data_clean['DE_NEED_RATIO'] =  all_data_clean['DE_NEED'] / all_data_clean["DE_PRODUCTION"] 

    
    return all_data_clean


def clean_data(all_data):
    
    all_data_clean = all_data.copy()
    
    all_data_clean['COUNTRY'] = all_data_clean['COUNTRY'].apply(lambda x: 0 if x =='FR' else 1) #one hot encoding
    
    #clean
    all_data_clean = clean_regression(all_data_clean)

    #remove correlated features
    all_data_clean = remove_features(all_data_clean)
    
    return all_data_clean
    
    
def split(all_data_clean):
    
    X_train = all_data_clean[all_data_clean['train'] == 1].drop(['train','TARGET','ID', 'DAY_ID'],axis=1)
    X_test = all_data_clean[all_data_clean['train'] == 0].drop(['train','TARGET','DAY_ID'],axis=1)
    id_test = X_test['ID']
    X_test = X_test.drop('ID',axis=1)
    y_train = all_data_clean[all_data_clean['train'] == 1]['TARGET']

    return X_train, y_train, X_test, id_test
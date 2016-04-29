import time

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from sklearn.grid_search import GridSearchCV
import re
import random
random.seed(2020)


_path = '/home/rajdeep.banerjee/test/'#'/opt/PaymentGatewayRouting/misc/K_HomeDepotSrchRel/'

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df_all= pd.read_csv(_path+'input/'+'df_all_new_feat2.csv', encoding="ISO-8859-1")
srch_vec = pd.read_csv(_path+'input/'+'new_search_term_word2vecs.csv').fillna(0.0)
pdt_ttl_vec = pd.read_csv(_path+'input/'+'product_title_word2vecs.csv').fillna(0.0)
pdt_desc_vec = pd.read_csv(_path+'input/'+'product_description_word2vecs.csv').fillna(0.0)


srch_vec=srch_vec.as_matrix(columns=[srch_vec.columns[:300]])
pdt_ttl_vec=pdt_ttl_vec.as_matrix(columns=[pdt_ttl_vec.columns[:300]])
pdt_desc_vec=pdt_desc_vec.as_matrix(columns=[pdt_desc_vec.columns[:300]])

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity

dst_srch_ttl1 = np.zeros(srch_vec.shape[0])
for i in range(srch_vec.shape[0]):
    d1 = srch_vec[i,:]
    d2 = pdt_ttl_vec[i,:]
    dst_srch_ttl1[i] = cosine_similarity(d1,d2)
    
dst_srch_desc1 = np.zeros(srch_vec.shape[0])
for i in range(srch_vec.shape[0]):
    d1 = srch_vec[i,:]
    d2 = pdt_desc_vec[i,:]
    dst_srch_desc1[i] = cosine_similarity(d1,d2)
    
dst_ttl_desc1 = np.zeros(srch_vec.shape[0])
for i in range(srch_vec.shape[0]):
    d1 = pdt_ttl_vec[i,:]
    d2 = pdt_desc_vec[i,:]
    dst_srch_desc1[i] = cosine_similarity(d1,d2)


svd = TruncatedSVD(n_components=30, random_state = 2016)

srch_vec=svd.fit_transform(srch_vec)
pdt_ttl_vec=svd.fit_transform(pdt_ttl_vec)
pdt_desc_vec=svd.fit_transform(pdt_desc_vec)

srch_vec=pd.DataFrame(srch_vec, columns = ['srch_vec_'+str(i) for i in range(srch_vec.shape[1])])
pdt_ttl_vec=pd.DataFrame(pdt_ttl_vec, columns = ['ttl_vec_'+str(i) for i in range(pdt_ttl_vec.shape[1])])
pdt_desc_vec=pd.DataFrame(pdt_desc_vec, columns = ['desc_vec_'+str(i) for i in range(pdt_desc_vec.shape[1])])

id = list(df_all['id'])
srch_vec['id']=id
pdt_ttl_vec['id']=id
pdt_desc_vec['id']=id

df_all=pd.merge(df_all, srch_vec, how='left', on ='id')
df_all=pd.merge(df_all, pdt_ttl_vec, how='left', on ='id')
df_all=pd.merge(df_all, pdt_desc_vec, how='left', on ='id')

df_all['dst_srch_ttl1']=dst_srch_ttl1
df_all['dst_srch_desc1']=dst_srch_desc1
df_all['dst_ttl_desc1']=dst_ttl_desc1


cols = list(df_all.select_dtypes(include=['object']).columns)
#[u'product_title', u'search_term', u'product_description', u'brand', u'material',
# u'colour', u'new_search_term', u'product_info', u'attr']

df_all1=df_all.drop(cols,1)
df_all1.to_csv(_path+'input/df_all_new_feat3.csv',index=False)

#Training
df_all= pd.read_csv(_path+'input/'+'df_all_new_feat3.csv', encoding="ISO-8859-1")

df_val=df_all[df_all['relevance'].isnull()]
df_train=df_all[~df_all['relevance'].isnull()]
id_val = df_val['id']
y_train = df_train['relevance'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.3, random_state=1234)

X_train = X_train.drop(['id','relevance'],axis=1)
X_test = X_test.drop(['id','relevance'],axis=1)

import xgboost as xgb
from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def RMSE(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

def run_test_func(params):
    n_estimators, learning_rate, max_depth, min_child_weight, subsample, gamma, colsample_bytree = params
    n_estimators=int(n_estimators)
    print params
    clf = xgb.XGBRegressor(silent=False, objective="reg:linear", nthread=1,n_estimators= n_estimators,
                                 learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,   
                                subsample=subsample, gamma=gamma,colsample_bytree=colsample_bytree)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print y_test[1], y_pred[1]
    rmse = RMSE( y_test, y_pred )
    #print "RMSE:", rmse
    return rmse     
    
    
def optimize(trials):
    space = (
             hp.quniform('n_estimators', 300,1000, 10),
             hp.quniform('learning_rate', 0.1, 0.5, 0.05),
             hp.quniform('max_depth', 3, 10, 1),
             hp.quniform('min_child_weight', 1, 6, 1),
             hp.quniform('subsample', 0.5, 1, 0.05),
             hp.quniform('gamma', 0.5, 1, 0.05),
             hp.quniform('colsample_bytree', 0.5, 1, 0.05)
             )
    best = fmin(run_test_func, space, algo=tpe.suggest, trials=trials, max_evals=10)
    print best      
       

trials = Trials()
optimize(trials)
trials.best_trial  

#{'colsample_bytree': 0.75, 'learning_rate': 0.15000000000000002, 'min_child_weight': 4.0, 'n_estimators': 340.0, 
#'subsample': 0.75, 'max_depth': 4.0, 'gamma': 0.9}
#>>> trials.best_trial   
#{'refresh_time': None, 'book_time': None, 'misc': {'tid': 2, 'idxs': {'colsample_bytree': [2], 'learning_rate': [2], 'min_child_weight': [2], 'n_estimators': [2], 'subsample': [2], 'max_depth': [2], 'gamma': [2]}, 'cmd': ('domain_attachment', 'FMinIter_Domain'), 'vals': {'colsample_bytree': [0.75], 'learning_rate': [0.15000000000000002], 'min_child_weight': [4.0], 'n_estimators': [340.0], 'subsample': [0.75], 'max_depth': [4.0], 'gamma': [0.9]}, 'workdir': None}, 'state': 2, 'tid': 2, 'exp_key': None, 'version': 0,
# 'result': {'status': 'ok', 'loss': 0.46093954224126193}, 'owner': None, 'spec': None}               
##########################################################################################################

df_val=df_all[df_all['relevance'].isnull()]
df_train=df_all[~df_all['relevance'].isnull()]
id_val = df_val['id']
y_train = df_train['relevance'].values

X_train = df_train.drop(['id','relevance'],axis=1)
X_test = df_val.drop(['id','relevance'],axis=1)



clf = xgb.XGBRegressor(silent=False, objective="reg:linear",n_estimators= 340,
                                 learning_rate=0.15, max_depth=4, min_child_weight=4,   
                                subsample=0.75, gamma=0.9,colsample_bytree=0.75)
from sklearn import cross_validation
from sklearn import metrics

t0=time.time()
scores = cross_validation.cross_val_score(clf, X_train, y_train,cv=3, scoring="mean_squared_error",n_jobs=-1)
print scores
print 'Cross Val time taken:', (time.time()-t0)/60.0, 'minutes'

clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

min_y_pred = min(y_pred)
max_y_pred = max(y_pred)
min_y_train = min(y_train)
max_y_train = max(y_train)
print(min_y_pred, max_y_pred, min_y_train, max_y_train)
for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0
    #y_pred[i] = min_y_train + (((y_pred[i] - min_y_pred)/(max_y_pred - min_y_pred))*(max_y_train - min_y_train))
 
id_test = df_val['id']   
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(_path+'xgb_vec1.csv',index=False)

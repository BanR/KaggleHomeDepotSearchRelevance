import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
from nltk.metrics import edit_distance
from sklearn.grid_search import GridSearchCV
import re
import random
random.seed(2020)
import xgboost as xgb

_path = '/home/kaggle/'

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df_all= pd.read_csv(_path+'input/'+'df_all_new_feat2.csv', encoding="ISO-8859-1")
#srch_vec = pd.read_csv(_path+'input/'+'new_search_term_word2vecs.csv')
#pdt_ttl_vec = pd.read_csv(_path+'input/'+'product_title_word2vecs.csv')

#df_all = pd.merge(df_all1, srch_vec, how='left', on='id')
#df_all = pd.merge(df_all, pdt_ttl_vec, how='left', on='id')
#del srch_vec, pdt_ttl_vec


import sys
reload (sys)
sys.setdefaultencoding('utf8')


df_test=df_all[df_all['relevance'].isnull()]
df_train=df_all[~df_all['relevance'].isnull()]
#df_train = df_all.iloc[:num_train]
#df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values

#X_train = df_train.drop(['id','search_term','relevance','product_info','attr','attr1','attr2','attr3','attr4','attr5'],axis=1)
#X_test = df_test.drop(['id','search_term','relevance','product_info','attr','attr1','attr2','attr3','attr4','attr5'],axis=1)

X_train = df_train.drop(['id','search_term','relevance','product_info','attr'],axis=1)
X_test = df_test.drop(['id','search_term','relevance','product_info','attr'],axis=1)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['new_search_term','product_title','product_description','brand','material','colour']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)
        
#xgtrain = xgb.DMatrix(train.values, y_train.values)
#xgtest = xgb.DMatrix(test.values)
#{'colsample_bytree': 1.0, 'learning_rate': 0.4, 'min_child_weight': 2.0, 
#'n_estimators': 358.0, 'subsample': 0.8500000000000001, 'max_depth': 8.0, 'gamma': 0.7000000000000001}



xgb_model = xgb.XGBRegressor(learning_rate=0.4, silent=False, objective="reg:linear", nthread=1, gamma=0.7, min_child_weight=2, max_delta_step=0,
                 subsample=0.85, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, seed=0, missing=None)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=100, random_state = 2016)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='new_search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', TruncatedSVD(n_components=10, random_state = 2016))])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', TruncatedSVD(n_components=10, random_state = 2016))]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.25,
                        'txt4': 0.5,
                        'txt5': 0.5
                        },
                #n_jobs = -1
                )), 
        ('xgb_model', xgb_model)])
#{'colsample_bytree': 0.55, 'learning_rate': 0.1, 'min_child_weight': 1.0, 'n_estimators': 230.0, 'subsample': 0.8500000000000001, 'max_depth': 5.0, 'gamma': 0.65}
param_grid = {'xgb_model__learning_rate':[0.1],'xgb_model__gamma':[0.65,0.75],'xgb_model__max_depth': [3,5], 'xgb_model__n_estimators': [230,400]	,'xgb_model__colsample_bytree':[0.55],'xgb_model__min_child_weight':[1.0],'xgb_model__subsample':[0.85]}

model = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -2, cv = 3, verbose = 10, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)

y_pred = model.predict(X_test)

#print(len(y_pred))
#pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('submission_before.csv',index=False)
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
    
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(_path+'subm/'+'submission_xgb_7.csv',index=False)
print("--- Training & Testing: %s minutes ---" % round(((time.time() - start_time)/60),2))



#{'xgb_model__n_estimators': 400, 'xgb_model__max_depth': 3, 'xgb_model__gamma': 0.75, 'xgb_model__learning_rate': 0.1}
#>>> print("Best CV score:")
#Best CV score:
#>>> print(model.best_score_)
#-0.468579014083
#
#
###################################
#xgb_model = xgb.XGBRegressor(learning_rate=0.1, silent=False, objective="reg:linear", nthread=1, gamma=0, min_child_weight=4, max_delta_step=0,
#                 subsample=1 ,colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#                 base_score=0.5, seed=0, missing=None)
#                 
#param_grid = {'xgb_model__max_depth': [8], 'xgb_model__n_estimators': [250,300]}
#Best parameters found by grid search:
#>>> print(model.best_params_)
#{'xgb_model__n_estimators': 250, 'xgb_model__max_depth': 8}
#>>> print("Best CV score:")
#Best CV score:
#>>> print(model.best_score_)
#-0.476383861638
                 

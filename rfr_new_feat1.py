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
from sklearn.ensemble import RandomForestRegressor

_path = '/home/rajdeep.banerjee/test/'#'/opt/PaymentGatewayRouting/misc/K_HomeDepotSrchRel/'

def fmean_squared_error(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_

RMSE  = make_scorer(fmean_squared_error, greater_is_better=False)


df_all1 = pd.read_csv(_path+'input/'+'df_all_new_feat.csv', encoding="ISO-8859-1")
srch_vec = pd.read_csv(_path+'input/'+'new_search_term_word2vecs.csv')
pdt_ttl_vec = pd.read_csv(_path+'input/'+'product_title_word2vecs.csv')

df_all = pd.merge(df_all1, srch_vec, how='left', on='id')
df_all = pd.merge(df_all, pdt_ttl_vec, how='left', on='id')
del srch_vec, pdt_ttl_vec


import sys
reload (sys)
sys.setdefaultencoding('utf8')


df_test=df_all[df_all['relevance'].isnull()]
df_train=df_all[~df_all['relevance'].isnull()]
df_test=df_test.fillna(0)
df_train=df_train.fillna(0)
#df_train = df_all.iloc[:num_train]
#df_test = df_all.iloc[num_train:]
id_test = df_test['id']
y_train = df_train['relevance'].values

X_train = df_train.drop(['id','search_term','relevance','product_info','attr'],axis=1)
X_test = df_test.drop(['id','search_term','relevance','product_info','attr'],axis=1)


class cust_regression_vals(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['new_search_term','product_title','product_description','brand','material']
        hd_searches = hd_searches.drop(d_col_drops,axis=1).values
        return hd_searches

class cust_txt_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key].apply(str)
        
rfr = RandomForestRegressor(n_estimators = 500, n_jobs = 1, random_state = 2020, verbose = 1)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
tsvd = TruncatedSVD(n_components=100, random_state = 2020)
clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),  
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='new_search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)])),
                        ('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', tsvd)])),
                        ('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', tsvd)]))
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
        ('rfr', rfr)])
param_grid = {'rfr__max_depth': [20,25,30],'rfr__n_estimators':[1000,1200,1500]}
print param_grid
model = GridSearchCV(estimator = clf, param_grid = param_grid, n_jobs = -2, cv = 3, verbose = 20, scoring=RMSE)
model.fit(X_train, y_train)

print("Best parameters found by grid search:")
print(model.best_params_)
print("Best CV score:")
print(model.best_score_)
#print(model.best_score_ + 0.47003199274)

y_pred = model.predict(X_test)

for i in range(len(y_pred)):
    if y_pred[i]<1.0:
        y_pred[i] = 1.0
    if y_pred[i]>3.0:
        y_pred[i] = 3.0
        
pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv(_path+'subm/'+'rfr_bm3.csv',index=False)        
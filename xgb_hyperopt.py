import numpy as np
import pandas as pd
from time import time
import csv
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



path='/home/kaggle/'

df_all = pd.read_csv(path+'df_all_new_feat2.csv', encoding="ISO-8859-1")


df_val=df_all[df_all['relevance'].isnull()]
df_train=df_all[~df_all['relevance'].isnull()]
id_val = df_val['id']
y_train = df_train['relevance'].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_train, y_train, test_size=0.3, random_state=1234)


X_train = X_train.drop(['id','search_term','relevance','product_info','attr'],axis=1)
X_test = X_test.drop(['id','search_term','relevance','product_info','attr'],axis=1)

import sys
reload (sys)
sys.setdefaultencoding('utf8')

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

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

#run_counter = 0
#ef run_wrapper( params ):
#	global run_counter
#	global o_f
#	run_counter += 1
#	print "run", run_counter
#	s = time()
#	rmse = run_test_func( params )
#	#print
#	print "RMSE:", rmse
#	print "elapsed: {}s \n".format( int( round( time() - s )))
#	writer.writerow( [ rmse ] + list( params ))
#	o_f.flush()
#	return rmse

#XGB Tuning
def run_test_func(params):
    n_estimators, learning_rate, max_depth, min_child_weight, subsample, gamma, colsample_bytree = params
    n_estimators=int(n_estimators)
    print params
    xgb_model = xgb.XGBRegressor(silent=False, objective="reg:linear", nthread=1,n_estimators= n_estimators,
                                 learning_rate=learning_rate, max_depth=max_depth, min_child_weight=min_child_weight,   
                                subsample=subsample, gamma=gamma,colsample_bytree=colsample_bytree)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=100)
    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='new_search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)]))
                        #('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', TruncatedSVD(n_components=5))])),
                        #('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', TruncatedSVD(n_components=5))]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.25
                        #'txt4': 0.5,
                        #'txt5': 0.5
                        },
                #n_jobs = -1
                )), 
        ('xgb_model', xgb_model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print y_test[1], y_pred[1]
    rmse = RMSE( y_test, y_pred )
    #print "RMSE:", rmse
    return rmse

def RMSE(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


def optimize(trials):
    space = (
             hp.quniform('n_estimators', 200, 500, 10),
             hp.quniform('learning_rate', 0.1, 0.5, 0.05),
             hp.quniform('max_depth', 3, 10, 1),
             hp.quniform('min_child_weight', 1, 6, 1),
             hp.quniform('subsample', 0.5, 1, 0.05),
             hp.quniform('gamma', 0.5, 1, 0.05),
             hp.quniform('colsample_bytree', 0.5, 1, 0.05)
             )
    best = fmin(run_test_func, space, algo=tpe.suggest, trials=trials, max_evals=100)
    print best      
       

trials = Trials()
optimize(trials)

trials.best_trial
#{'refresh_time': None, 'book_time': None, 'misc': {'tid': 17, 'idxs': {'colsample_bytree': [17], 'learning_rate': [17],
# 'min_child_weight': [17], 'n_estimators': [17], 'subsample': [17], 'max_depth': [17], 'gamma': [17]}, 
# 'cmd': ('domain_attachment', 'FMinIter_Domain'), 
# 'vals': {'colsample_bytree': [0.55], 'learning_rate': [0.1], 'min_child_weight': [1.0], 'n_estimators': [230.0], 
# 'subsample': [0.8500000000000001], 'max_depth': [5.0], 'gamma': [0.65]}, 'workdir': None}, 'state': 2, 'tid': 17, 
# 'exp_key': None, 'version': 0, 'result': {'status': 'ok', 'loss': 0.48326965088314083}, 'owner': None, 'spec': None}


#trials.results
#[{'status': 'ok', 'loss': 0.5287705564805382}, {'status': 'ok', 'loss': 0.5167556015339757}, {'status': 'ok', 'loss': 0.4848251983308158}, {'status': 'ok', 'loss': 0.5078150929047329}, {'status': 'ok', 'loss': 0.5216752606579843}, {'status': 'ok', 'loss': 0.48629451649469646}, {'status': 'ok', 'loss': 0.5111041814224146}, {'status': 'ok', 'loss': 0.49635695807263647}, {'status': 'ok', 'loss': 0.5344371981226101}, {'status': 'ok', 'loss': 0.5234257116541885}, {'status': 'ok', 'loss': 0.49685922532977556}, {'status': 'ok', 'loss': 0.519963480893974}, {'status': 'ok', 'loss': 0.5374740204429276}, {'status': 'ok', 'loss': 0.4885870189608335}, {'status': 'ok', 'loss': 0.5101757733465687}, {'status': 'ok', 'loss': 0.5537791392513546}, {'status': 'ok', 'loss': 0.5044190529289339}, {'status': 'ok', 'loss': 0.48326965088314083}, {'status': 'ok', 'loss': 0.5145898170045293}, {'status': 'ok', 'loss': 0.499322178690165}, {'status': 'ok', 'loss': 0.4842726492685017}, {'status': 'ok', 'loss': 0.484998007913103}, {'status': 'ok', 'loss': 0.4851237182006996}, {'status': 'ok', 'loss': 0.48734865606082406}, {'status': 'ok', 'loss': 0.4846627588729656}, {'status': 'ok', 'loss': 0.5032828101616209}, {'status': 'ok', 'loss': 0.4864237738493829}, {'status': 'ok', 'loss': 0.4881871005472381}, {'status': 'ok', 'loss': 0.5044708868918563}, {'status': 'ok', 'loss': 0.4896497653606583}, {'status': 'ok', 'loss': 0.48447568668134106}, {'status': 'ok', 'loss': 0.5285960903998862}, {'status': 'ok', 'loss': 0.4848748285313814}, {'status': 'ok', 'loss': 0.4926642604088152}, {'status': 'ok', 'loss': 0.4855576665802113}, {'status': 'ok', 'loss': 0.5114337640617251}, {'status': 'ok', 'loss': 0.4981724016325844}, {'status': 'ok', 'loss': 0.49090528191967375}, {'status': 'ok', 'loss': 0.5119716400563775}, {'status': 'ok', 'loss': 0.504943508167469}, {'status': 'ok', 'loss': 0.4848651774744163}, {'status': 'ok', 'loss': 0.4913551647864685}, {'status': 'ok', 'loss': 0.49060676972440476}, {'status': 'ok', 'loss': 0.4982801224297972}, {'status': 'ok', 'loss': 0.5043368095929373}, {'status': 'ok', 'loss': 0.48480226706260743}, {'status': 'ok', 'loss': 0.5230170094339189}, {'status': 'ok', 'loss': 0.48646488649260555}, {'status': 'ok', 'loss': 0.4858642053757507}, {'status': 'ok', 'loss': 0.5130619504159252}, {'status': 'ok', 'loss': 0.4983455620218279}, {'status': 'ok', 'loss': 0.4835150434751544}, {'status': 'ok', 'loss': 0.5225855276335454}, {'status': 'ok', 'loss': 0.4874537391713808}, {'status': 'ok', 'loss': 0.48761238853376637}, {'status': 'ok', 'loss': 0.48536697655766115}, {'status': 'ok', 'loss': 0.5069379029321908}, {'status': 'ok', 'loss': 0.514944342109151}, {'status': 'ok', 'loss': 0.501257373630492}, {'status': 'ok', 'loss': 0.48354840464739896}, {'status': 'ok', 'loss': 0.48895079180315393}, {'status': 'ok', 'loss': 0.486414390250732}, {'status': 'ok', 'loss': 0.484316521893438}, {'status': 'ok', 'loss': 0.5324038137815281}, {'status': 'ok', 'loss': 0.4841519889537018}, {'status': 'ok', 'loss': 0.4839038709646751}, {'status': 'ok', 'loss': 0.484417089561097}, {'status': 'ok', 'loss': 0.487357740883246}, {'status': 'ok', 'loss': 0.4847183112278339}, {'status': 'ok', 'loss': 0.48574164312022955}, {'status': 'ok', 'loss': 0.4922738586899265}, {'status': 'ok', 'loss': 0.48838640831872987}, {'status': 'ok', 'loss': 0.48599104751504596}, {'status': 'ok', 'loss': 0.4856893338508745}, {'status': 'ok', 'loss': 0.4858897803210282}, {'status': 'ok', 'loss': 0.4905496365691216}, {'status': 'ok', 'loss': 0.48610915572273217}, {'status': 'ok', 'loss': 0.48833823613192806}, {'status': 'ok', 'loss': 0.48416647214790315}, {'status': 'ok', 'loss': 0.4950493578519844}, {'status': 'ok', 'loss': 0.5065273534941873}, {'status': 'ok', 'loss': 0.48598908804738894}, {'status': 'ok', 'loss': 0.484964430331251}, {'status': 'ok', 'loss': 0.49022826974367995}, {'status': 'ok', 'loss': 0.519193267683132}, {'status': 'ok', 'loss': 0.4848455552443866}, {'status': 'ok', 'loss': 0.48371746021282896}, {'status': 'ok', 'loss': 0.5112306979162786}, {'status': 'ok', 'loss': 0.498872889185528}, {'status': 'ok', 'loss': 0.4881413064996127}, {'status': 'ok', 'loss': 0.4843692910104079}, {'status': 'ok', 'loss': 0.4889006990105608}, {'status': 'ok', 'loss': 0.4965634486363468}, {'status': 'ok', 'loss': 0.48695177967276393}, {'status': 'ok', 'loss': 0.5134485187016947}, {'status': 'ok', 'loss': 0.5062992983330228}, {'status': 'ok', 'loss': 0.49843124434815955}, {'status': 'ok', 'loss': 0.4837740898935797}, {'status': 'ok', 'loss': 0.4881620592434986}, {'status': 'ok', 'loss': 0.4935786149453978}]


#RfR tuning
from sklearn.ensemble import RandomForestRegressor

def run_test_func(params):
    max_depth, n_estimators = params
    n_estimators=int(n_estimators)
    print params
    rfr = RandomForestRegressor(max_depth=max_depth, n_estimators = n_estimators, verbose = 1)
    tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
    tsvd = TruncatedSVD(n_components=100)
    clf = pipeline.Pipeline([
        ('union', FeatureUnion(
                    transformer_list = [
                        ('cst',  cust_regression_vals()),
                        ('txt1', pipeline.Pipeline([('s1', cust_txt_col(key='new_search_term')), ('tfidf1', tfidf), ('tsvd1', tsvd)])),
                        ('txt2', pipeline.Pipeline([('s2', cust_txt_col(key='product_title')), ('tfidf2', tfidf), ('tsvd2', tsvd)])),
                        ('txt3', pipeline.Pipeline([('s3', cust_txt_col(key='product_description')), ('tfidf3', tfidf), ('tsvd3', tsvd)]))
                        #('txt4', pipeline.Pipeline([('s4', cust_txt_col(key='brand')), ('tfidf4', tfidf), ('tsvd4', TruncatedSVD(n_components=5))])),
                        #('txt5', pipeline.Pipeline([('s5', cust_txt_col(key='material')), ('tfidf5', tfidf), ('tsvd5', TruncatedSVD(n_components=5))]))
                        ],
                    transformer_weights = {
                        'cst': 1.0,
                        'txt1': 0.5,
                        'txt2': 0.25,
                        'txt3': 0.25
                        #'txt4': 0.5,
                        #'txt5': 0.5
                        },
                #n_jobs = -1
                )), 
        ('rfr_model', rfr)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    #print y_test[1], y_pred[1]
    rmse = RMSE( y_test, y_pred )
    #print "RMSE:", rmse
    return rmse
    
def RMSE(ground_truth, predictions):
    fmean_squared_error_ = mean_squared_error(ground_truth, predictions)**0.5
    return fmean_squared_error_


def optimize(trials):
    space = (
             #hp.quniform('max_features', 10, 50, 10),
             hp.quniform('max_depth', 3,30,1),
             hp.quniform('n_estimators', 100,1500,10)
             )
    best = fmin(run_test_func, space, algo=tpe.suggest, trials=trials, max_evals=20)
    print best      
       

trials = Trials()
optimize(trials)

trials.best_trial

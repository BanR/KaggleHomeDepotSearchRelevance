# -*- coding: utf-8 -*-
import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.snowball import SnowballStemmer 
stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(2016)


import gensim
model = gensim.models.Word2Vec.load_word2vec_format('/home/GoogleNews-vectors-negative300.bin.gz', binary=True)
print 'Model loaded'

_path = '/home/kaggle/'
df_train = pd.read_csv(_path+'/input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(_path+'/input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(_path+'/input/product_descriptions.csv',encoding="ISO-8859-1")
df_attr = pd.read_csv(_path+'/input/attributes.csv',encoding="ISO-8859-1")
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_material = df_attr[df_attr.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})

df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_material1 = df_material.drop_duplicates(['product_uid'])
df_all = pd.merge(df_all, df_material1, how='left', on='product_uid')
print df_all.shape

#correct search term 
with open(_path+'/input/correct_pdt_search.txt', 'r') as f:
    data = f.read()

import ast
corr_pdts = ast.literal_eval(data)

corr_pdt_list = list(corr_pdts.keys())
df_all['new_search_term'] = df_all['search_term'].map(lambda x: corr_pdts[x] if x in corr_pdt_list else x)


import sys
reload (sys)
sys.setdefaultencoding('ISO-8859-1')

print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def str_stem(s): 
    if isinstance(s, str):
        #print 'its a str'
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
        s = s.replace("°"," degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
        s = s.replace(" v "," volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)
        s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([z for z in s.split(" ") if z not in stop_w])
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        #s = (" ").join([correct(z) for z in s.split(" ")])  #spell correcter
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.lower()
        s = s.replace("toliet","toilet")
        s = s.replace("airconditioner","air conditioner")
        s = s.replace("vinal","vinyl")
        s = s.replace("vynal","vinyl")
        s = s.replace("skill","skil")
        s = s.replace("snowbl","snow bl")
        s = s.replace("plexigla","plexi gla")
        s = s.replace("rustoleum","rust-oleum")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless","whirlpool stainless")
        s = s.replace("glas","glass")
        s = s.replace("fiberglas","fiberglass")
        s = s.replace("poplar","popular")
        s = s.replace("engineereed","engineered")
        s = s.replace("galvinized","galvanized")  
        s = s.replace("polyethelene","polyethylene")  
        s = s.replace("polyethelyne","polyethylene") 
        s = s.replace("polyethelyne","polyethylene") 
        s = s.replace("polypropelene","polypropylene") 
        s = s.replace("sainless","stainless") 
        s = s.replace("denisty","density") 
        s = s.replace("high density fiberboard hdf ","high density fiber board") 
        s = s.replace("hdf ","high density fiber board") 
        s = s.replace("palstic ","plastic") 
        s = s.replace("closetmaid ","closet maid") 
        return s
    else:
        return "null"
        
print ("String clean ups...")        
df_all['new_search_term'] = df_all['new_search_term'].astype(str).map(lambda x:str_stem(x))#comment out corrector in str_stem function
df_all['product_title'] = df_all['product_title'].astype(str).map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].astype(str).map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].astype(str).map(lambda x:str_stem(x))
df_all['brand']  = df_all['brand'].fillna('unbrand')
df_all['brand'] = np.where(df_all['brand'].isin (['n a ','na',' na','nan']),'unbrand',df_all['brand'])

df_all.to_csv(_path+'input/df_all_vect.csv',index=False)

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    review_text = re.sub("[^a-zA-Z]"," ", review)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return(words)


# Function to average all of the word vectors in a given paragraph
def makeFeatureVec(words, model, num_features):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    nwords = 0.
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    featureVec = np.divide(featureVec,nwords)
    return featureVec

# Given a set of reviews (each one a list of words), calculate 
# the average feature vector for each one and return a 2D numpy array 
def getAvgFeatureVecs(reviews, model, num_features):
    # Initialize a counter
    counter = 0.
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%100. == 0.:
           print "done %d of %d" % (counter, len(reviews))
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, num_features)
       #Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

# ****************************************************************
# Calculate average feature vectors 

num_style_desc = df_all.shape[0]
for col in ['new_search_term','product_title','product_description']:
    #print col
    cleantxt=[]
    for i in xrange( 0, num_style_desc ):
        if( (i+1)%10000== 0 ):
            print "vectors %d of %d\n" % ( i+1, num_style_desc )
        cleantxt.append(review_to_wordlist(df_all[col][i] ,remove_stopwords=False) )
    print col	
    vec = col+'_Vecs'
    num_features=300
    vec = getAvgFeatureVecs( cleantxt, model, num_features )
    print vec.shape
    df_vec = pd.DataFrame(vec,columns = ['vec_'+col+'_'+str(k) for k in range(num_features)])
    df_vec.to_csv(_path+col+'_word2vecs.csv',index=False)
    

    

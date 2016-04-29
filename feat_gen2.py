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

_path = '/home/rajdeep.banerjee/test/'#/Users/8199/Documents/ml/Kaggle/Kaggle_HomeDepot/'  #'/opt/PaymentGatewayRouting/misc/K_HomeDepotSrchRel/'
#######################
df_train = pd.read_csv(_path+'/input/train.csv', encoding="ISO-8859-1")
df_test = pd.read_csv(_path+'/input/test.csv', encoding="ISO-8859-1")
df_pro_desc = pd.read_csv(_path+'/input/product_descriptions.csv',encoding="ISO-8859-1")


df_attr = pd.read_csv(_path+'/input/attributes.csv',encoding="ISO-8859-1")
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"}).drop_duplicates(['product_uid'])
df_material = df_attr[df_attr.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"}).drop_duplicates(['product_uid'])

df_color = df_attr[df_attr.name.isin (['Color Family','Color/Finish','Color','Color/Finish Family','Fixture Color/Finish'])][["product_uid", "value"]].rename(columns={"value": "colour"}).drop_duplicates(['product_uid'])

df_width_msr =  df_attr[df_attr.name.isin (['Product Width (in.)','Assembled Width (in.)'])][["product_uid", "value"]].rename(columns={"value": "width_msr"}).drop_duplicates(['product_uid'])
df_height_msr =  df_attr[df_attr.name.isin (['Product Height (in.)','Assembled Height (in.)'])][["product_uid", "value"]].rename(columns={"value": "height_msr"}).drop_duplicates(['product_uid'])
df_length_msr =  df_attr[df_attr.name.isin (['Product Length (in.)'])][["product_uid", "value"]].rename(columns={"value": "length_msr"}).drop_duplicates(['product_uid'])
df_depth_msr =  df_attr[df_attr.name.isin (['Product Depth (in.)','Assembled Depth (in.)'])][["product_uid", "value"]].rename(columns={"value": "depth_msr"}).drop_duplicates(['product_uid'])
df_weight_msr =  df_attr[df_attr.name.isin (['Product Weight (lb.)'])][["product_uid", "value"]].rename(columns={"value": "weight_msr"}).drop_duplicates(['product_uid'])

df_certifications = df_attr[df_attr.name.isin (['Certifications and Listings'])][["product_uid", "value"]].rename(columns={"value": "certifications"}).drop_duplicates(['product_uid'])
df_certifications['certification_flag'] = np.where(df_certifications['certifications']=='No Certifications or Listings',0,1)

df_energycertifications = df_attr[df_attr.name.isin (['ENERGY STAR Certified'])][["product_uid", "value"]].rename(columns={"value": "energy_certifications"}).drop_duplicates(['product_uid'])
df_energycertifications['engy_cert_flag'] = np.where(df_energycertifications['energy_certifications']=='No',0,1)

df_in_outdoor = df_attr[df_attr.name.isin (['Indoor/Outdoor'])][["product_uid", "value"]].rename(columns={"value": "in_outdoor"}).drop_duplicates(['product_uid'])
df_in_outdoor['is_indoor_flag'] = np.where(df_in_outdoor['in_outdoor'].isin (['Indoor','Indoor/Outdoor (Covered)']) , 1,0)

df_in_exterior = df_attr[df_attr.name.isin (['Interior/Exterior'])][["product_uid", "value"]].rename(columns={"value": "in_exterior"}).drop_duplicates(['product_uid'])
df_in_exterior['is_interior_flag'] = np.where(df_in_exterior['in_exterior']=='Exterior' , 0,1)

df_hdw_incl = df_attr[df_attr.name.isin (['Hardware Included'])][["product_uid", "value"]].rename(columns={"value": "hdw_incl"}).drop_duplicates(['product_uid'])
df_hdw_incl['hdw_incl_flag'] = np.where(df_hdw_incl['hdw_incl']=='Yes' , 1,0)

df_comm_resd_incl = df_attr[df_attr.name.isin (['Commercial / Residential'])][["product_uid", "value"]].rename(columns={"value": "comm_resd"}).drop_duplicates(['product_uid'])
df_comm_resd_incl['is_residential'] = np.where(df_comm_resd_incl['comm_resd']=='Commercial' ,0,1)

#merge data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')

df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
df_all = pd.merge(df_all, df_material, how='left', on='product_uid')
df_all = pd.merge(df_all, df_color, how='left', on='product_uid')
df_all = pd.merge(df_all, df_certifications, how='left', on='product_uid')
df_all = pd.merge(df_all, df_energycertifications, how='left', on='product_uid')
df_all = pd.merge(df_all, df_in_outdoor, how='left', on='product_uid')
df_all = pd.merge(df_all, df_in_exterior, how='left', on='product_uid')
df_all = pd.merge(df_all, df_hdw_incl, how='left', on='product_uid')
df_all = pd.merge(df_all, df_comm_resd_incl, how='left', on='product_uid')

df_all = pd.merge(df_all, df_width_msr, how='left', on='product_uid')
df_all = pd.merge(df_all, df_height_msr, how='left', on='product_uid')
df_all = pd.merge(df_all, df_length_msr, how='left', on='product_uid')
df_all = pd.merge(df_all, df_depth_msr, how='left', on='product_uid')
df_all = pd.merge(df_all, df_weight_msr, how='left', on='product_uid')


df_all['brand']=df_all['brand'].fillna('unbrand')
df_all['material']=df_all['material'].fillna('unknown')
df_all['colour']=df_all['colour'].fillna('unknown')
df_all['certification_flag']=df_all['certification_flag'].fillna(0)
df_all['engy_cert_flag']=df_all['engy_cert_flag'].fillna(0)
df_all['is_indoor_flag']=df_all['is_indoor_flag'].fillna(0)
df_all['is_interior_flag']=df_all['is_interior_flag'].fillna(0)
df_all['hdw_incl_flag']=df_all['hdw_incl_flag'].fillna(0)
df_all['is_residential']=df_all['is_residential'].fillna(0)

df_all['width_msr_flag']=np.where(df_all['width_msr'].isnull(),0,1)
df_all['height_msr_flag']=np.where(df_all['height_msr'].isnull(),0,1)
df_all['length_msr_flag']=np.where(df_all['length_msr'].isnull(),0,1)
df_all['depth_msr_flag']=np.where(df_all['depth_msr'].isnull(),0,1)
df_all['weight_msr_flag']=np.where(df_all['weight_msr'].isnull(),0,1)


df_all = df_all.drop(['certifications','energy_certifications','in_outdoor','in_exterior','hdw_incl','comm_resd','width_msr','height_msr','length_msr','depth_msr','weight_msr'],axis=1)
################################################################
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
    if isinstance(s, str) or isinstance(s, unicode):
        #s = s.lower()
        s = s.replace("deckover","deck over")
        s = s.replace("clouverleaf","cloverleaf")
        s = s.replace("grilll","gril")
        s = s.replace("rainbird","Rain Bird")
        s = s.replace("flexlock","flex lock")
        s = s.replace("ultratouch","ultra touch")
        s = s.replace("shelfes","shelfs")
        s = s.replace("shelfing","selfing")
        s = s.replace("kitchenen","kitchen")
        s = s.replace("boltless","bolt less")
        s = s.replace("drilll","drill")
        s = s.replace("ridgid","rigid")
        s = s.replace("emberglow","ember glow")
        s = s.replace("tooliet","toilet")
        s = s.replace("wiremesh","wire mesh")
        s = s.replace("prefinished","pre finished")
        s = s.replace("prehung","pre hung")
        s = s.replace("priceepfister","pfister")
        s = s.replace("ceilingg","ceiling")
        s = s.replace("trafficmaster","traffic master")
        s = s.replace("glacierr","glacier")
        s = s.replace("toolet","toilet")
        s = s.replace("watter","water")
        s = s.replace("rerefrigerator","refrigerator")
        s = s.replace("garagee","garage")
        s = s.replace("air condit","air conditioner")
        s = s.replace("deckpaint","deck paint")
        s = s.replace("drilll","drill")
        s = s.replace("flangee","flange")
        s = s.replace("chipaper","chipper")
        s = s.replace("cleanerr","cleaner")
        s = s.replace("fiberglasss","fiberglass")
        s = s.replace("paccage","package")
        s = s.replace("garagee","garage")
        s = s.replace("collegege","college")
        s = s.replace("whiteplywood","white plywood")
        s = s.replace("electronical","")
        s = s.replace("beadboard","breadboard")
        s = s.replace("porcelaine","porcelain")
        s = s.replace("treatedd","treated")
        s = s.replace("cedarsafe","cedar safe")
        s = s.replace("plywooden","plywood")
        s = s.replace("sundstrom","sandstorm")
        s = s.replace("sprkinler","sprinkler")
        s = s.replace("sinktop","sink top")
        s = s.replace("ourdoor","outdoor")
        s = s.replace("ultratouch","ultra touch")
        s = s.replace("decoart","deco art")
        s = s.replace("outdoorlounge","outdoor lounge")
        s = s.replace("outdoorfurniture","outdoor furniture")
        s = s.replace("pricepfister","pfister")
        s = s.replace("glaciar","glacier")
        s = s.replace("glacie","glacier")
        s = s.replace("tiolet","toilet")
        s = s.replace("handycap","handicap")
        s = s.replace("wayer","water")
        s = s.replace("undercounter","under counter")
        s = s.replace("basemetnt","basement")
        s = s.replace("rustollum","rustoleum")
        s = s.replace("heaterconditioner","air conditioner")
        s = s.replace("spliter","splitter")
        s = s.replace("berh","behr")
        s = s.replace("snow thower","snow blower")
        s = s.replace("powertool","power tool")
        s = s.replace("repir","repair")
        s = s.replace("condtioners","conditioners")
        s = s.replace("pannels","panels")
        s = s.replace("frostking","frost king")
        s = s.replace("flourescent","fluorescent")
        s = s.replace("closetmade","closetmaid")
        s = s.replace("repir","repair")
        s = s.replace("greecianmarble","grecian marble")
        s = s.replace("porcelin","porcelain")
        s = s.replace("flushmount","flush mount")
        s = s.replace("foof","foot")
        s = s.replace("incide","inside")
        s = s.replace("pedistal","pedestal")
        s = s.replace("miricale","miracle")
        s = s.replace("windos","windows")
        s = s.replace("closetmaid","closet maid")
        #s = s.replace("deckover","deck over")
        s = s.replace("aspiradora","aspirator")
        s = s.replace("bentgrass","bentgrass")
        s = s.replace("hindges","hinges")
        s = s.replace("hieght","height")
        s = s.replace("clab","clay")
        s = s.replace("procelian","porcelain")
        s = s.replace("wonderboard","wonder board")
        s = s.replace("backerboard","backer board")
        s = s.replace("flatbraces","flat braces")
        s = s.replace("cieling","ceiling")
        s = s.replace("ceadar","cedar")
        s = s.replace("cedart","cedar")
        s = s.replace("frontload","front load")
        s = s.replace("stcking","sticking")
        s = s.replace("barreir","barrier")
        s = s.replace("ajustable","adjustable")
        s = s.replace("sinnk","sink")
        s = s.replace("pedelal","pedestal")
        s = s.replace("undermount","under mounted")
        s = s.replace("suppll","supple")
        s = s.replace("conditionerionar","conditioner")
        s = s.replace("vynal","vinyl")
        s = s.replace("aluminun","aluminum")
        s = s.replace("installbay","install bay")
        s = s.replace("cermic","ceramic")
        s = s.replace("plastice","plastic")
        s = s.replace("wattsolar","watt solar")
        s = s.replace("glaciar","glacier")
        s = s.replace("toliet","toilet")
        s = s.replace("garageescape","garage escape")
        s = s.replace("alumanam","aluminum")
        s = s.replace("treate","treated")
        s = s.replace("weathershield","weather shield")
        s = s.replace("conditionerioners","conditioner")
        s = s.replace("heaterconditioner","conditioner")
        s = s.replace("vbration","vibration")
        s = s.replace("fencde","fence")
        s = s.replace("knoty","knotty")
        s = s.replace("untility","utility")
        s = s.replace("christmass","christmas")
        s = s.replace("garlend","garland")
        s = s.replace("ceilig","ceiling")
        s = s.replace("glaciar","glacier")
        s = s.replace("dcanvas","canvas")
        s = s.replace("vaccum","vacuum")
        s = s.replace("garge","garage")
        s = s.replace("ridiing","riding")
        s = s.replace("barreir","barrier")
        s = s.replace("keorsene","kerosene")
        s = s.replace("lanterun","lantern")
        s = s.replace("infered","infrared")
        s = s.replace("hardiboard","hardboard")
        s = s.replace("keorsene","kerosene")
        s = s.replace("sinnk","sink")
        s = s.replace("pedelal","pedaled")
        s = s.replace("hindged","hinged")
        s = s.replace("bateries","batteries")
        s = s.replace("undercabinet","under cabinet")
        s = s.replace("ceilig","ceiling")
        s = s.replace("extention","extension")
        s = s.replace("firepits","fire pit")
        s = s.replace("edsel","edsal")
        s = s.replace("aire acondicionado","air conditioner")
        s = s.replace("linoliuml","linoleum")
        s = s.replace("hagchet","hatchet")
        s = s.replace("steele","steel")
        s = s.replace("dimable","dimmable")
        s = s.replace("lithum","lithium")
        s = s.replace("rayoby","ryobi")
        s = s.replace("washerparts","washer kit")
        s = s.replace("lituim","lithium")
        s = s.replace("naturlas","naturals")
        s = s.replace("softners","softener")
        s = s.replace("doorsmoocher","doors moocher")
        s = s.replace("sofn","soft")
        s = s.replace("scaleblaster","scale blaster")
        s = s.replace("pressue","pressure")
        s = s.replace("paito","patio")
        s = s.replace("mandare","mandara")
        s = s.replace("scod","cod")
        s = s.replace("ddummy","dummy")
        s = s.replace("florcant","floor cant")
        s = s.replace("prunning","pruning")
        s = s.replace("enrty","enrty")
        s = s.replace("outdoorfurniture","outdoor furniture")
        s = s.replace("handtools","hand tools")
        s = s.replace("treate","treated")
        s = s.replace("wheelbarow","wheelbarrow")
        s = s.replace("hhigh","high")
        s = s.replace("accordian","accordion")
        s = s.replace("preature","pressure")
        s = s.replace("steqamers","steamers")
        s = s.replace("onda pressure","honda pressure")
        s = s.replace("insallation","insulation")
        s = s.replace("contracor","multi color")
        s = s.replace("stell","steel")
        s = s.replace("sjhelf","shelf")
        s = s.replace("ridiing","riding")
        s = s.replace("drils","drills")
        s = s.replace("planel","panel")
        s = s.replace("robi","ryobi")
        s = s.replace("ã¨_"," ")
        s = s.replace("swivrl","swirl")
        s = s.replace("enrty","entry")
        s = s.replace("paneks","panels")
        s = s.replace("floo shets","flooring sheets")
        s = s.replace("gazhose","gas hose")
        s = s.replace("artifical","artifical")
        s = s.replace("insullation","insulation")
        s = s.replace("peper","peper")
        s = s.replace("extention","extension")
        s = s.replace("insulaion","insulation")
        s = s.replace("insullation","insulation")
        s = s.replace("unsulation","insulation")
        s = s.replace("upholstry","upholstery")
        s = s.replace("medicien","medicine")
        s = s.replace("floorinh","flooring")
        s = s.replace("heavyduty","heavy duty")
        s = s.replace("hardsware","hardware")
        s = s.replace("traiter","trailer")
        s = s.replace("bathroon","bathroom")
        s = s.replace("tsnkless","tankless")
        s = s.replace("shoplight","shop light")
        s = s.replace("consertrated","concentrated")
        s = s.replace("zeroturn","zero turn")
        s = s.replace("vynik","vinyl")
        s = s.replace("aircondiioner","air conditioner")
        s = s.replace("plexy glass","plastic sheet")
        s = s.replace("accesory","accessory")
        s = s.replace("koolaroo","coolaroo")
        s = s.replace("uplihght","uplight")
        s = s.replace("edsel","edsal")
        s = s.replace("outdooor","outdoor")
        s = s.replace("pivotangle","pivot angle")
        s = s.replace("plasticl","plastic")
        s = s.replace("varigated","variegated")
        s = s.replace("basemetnt","basement")
        s = s.replace("cornor","corner")
        s = s.replace("plaers","pliers")
        s = s.replace("soundfroofing","roofing underlayment")
        s = s.replace("storeage","storage")
        s = s.replace("fountin","fountain")
        s = s.replace("extention","extension")
        s = s.replace("polyeurethane","polyurethane")
        s = s.replace("plastice","plastic")
        s = s.replace("tilees","tiles")
        s = s.replace("byefold","bi fold")
        s = s.replace("whirpool","whirlpool")
        s = s.replace("diining","dining")
        s = s.replace("connnector","connector")
        s = s.replace("woodebn","wooden")
        s = s.replace("celliling","ceiling")
        s = s.replace("waterroo","water roof")
        s = s.replace("sofn","softener")
        s = s.replace("softners","softener")
        s = s.replace("shoipping","shopping")
        s = s.replace("trollie","trolly")
        s = s.replace("shoplight","shop light")
        s = s.replace("diining","dining")
        s = s.replace("elongagated","elongated")
        s = s.replace("sjhelf","shelf")
        s = s.replace("polycarbonite","polycarbonate")
        s = s.replace("koolaroo","coolaroo")
        s = s.replace("galvinized","galvanized")
        s = s.replace("bakewarte","bakeware")
        s = s.replace("kennal","kennel")
        s = s.replace("elongagated","elongated")
        s = s.replace("tolet","toilet")
        s = s.replace("aspiradora","vacuum")
        s = s.replace("aluminium","aluminum")
        s = s.replace("laminet","laminate")
        s = s.replace("elecronic","electronic")
        s = s.replace("dedwalt","dewalt")
        s = s.replace("vaccuum","vacuum")
        s = s.replace("diining","dining")
        s = s.replace("insided","inside")
        s = s.replace("towbehind","tow behind")
        s = s.replace("kidie","kidde")
        s = s.replace("batterys","battery")
        s = s.replace("nutru","nutri")
        s = s.replace("kitchenfaucet","kitchen faucet")
        s = s.replace("kitcheen","kitchen")
        s = s.replace("toliet","toilet")
        #s = s.replace("airconditioner","air conditioner")
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
        s=s.replace("built-in", "builtin") 
		s=s.replace("feet.x", "feet x") 
		s=s.replace("plug-in", "plugin") 
		s=s.replace("all-pro", "allpro") 
		s=s.replace("all pro", "allpro") 
		s=s.replace("beach-pier", "beachpier") 
		s=s.replace("beach pier", "beachpier") 
		s=s.replace("k-4175f","k4175f") 
		s=s.replace("mobile home","mobilehome") 
		s=s.replace("mobile-home","mobilehome") 
		s=s.replace("double sided","doublesided") 
		s=s.replace("double-sided","doublesided") 
		s=s.replace("rj 14", "rj14") 
		s=s.replace("sun glasses", "sunglasses") 
		s=s.replace("waste basket", "wastebasket") 
		s=s.replace("waste  basket", "wastebasket") 
		s=s.replace("cast-iron", "castiron") 
		s=s.replace("cast iron", "castiron") 
		s=s.replace("self-stick", "selfstick") 
		s=s.replace("self stick", "selfstick") 
		s=s.replace("metro gris" , "metrogris") 
		s=s.replace("all in one", "allinone") 
		s=s.replace("all-in-one", "allinone") 
		s=s.replace("waste basket", "wastebasket") 
		s=s.replace("waste  basket", "wastebasket") 
		s=s.replace("whisper ceiling","whisperceiling") 
		s=s.replace("byefold" , "bifold") 
		s=s.replace("byfold" , "bifold") 
		s=s.replace("bi fold" , "bifold") 
		s=s.replace("bi-fold" , "bifold") 
		s=s.replace("candel bra" , "candelabra") 
		s=s.replace("candle bra" , "candelabra") 
		s=s.replace("come along" , "comealong") 
		s=s.replace("come-along" , "comealong") 
		s=s.replace("dead blow" , "deadblow") 
		s=s.replace("dead-blow" , "deadblow") 
		s=s.replace("deco strip" , "decostrip") 
		s=s.replace("deco-strip" , "decostrip") 
		s=s.replace("door guard" , "doorguard") 
		s=s.replace("door-guard" , "doorguard") 
		s=s.replace("doorsguard" , "doorguard") 
		s=s.replace("doorshinge" , "doorhinge") 
		s=s.replace("door hinge" , "doorhinge") 
		s=s.replace("door-hinge" , "doorhinge") 
		s=s.replace("doorsfold" , "doorfold") 
		s=s.replace("door-fold" , "doorfold") 
		s=s.replace("door fold" , "doorfold") 
		s=s.replace("door spring" , "doorspring") 
		s=s.replace("door-spring" , "doorspring") 
		s=s.replace("doorsscreen" , "doorscreen") 
		s=s.replace("doors screen" , "doorscreen") 
		s=s.replace("door-screen" , "doorscreen") 
		s=s.replace("door screen" , "doorscreen") 
		s=s.replace("double sided" , "doublesided") 
		s=s.replace("double-sided" , "doublesided") 
		s=s.replace("drop cloth" , "dropcloth") 
		s=s.replace("drop-cloth" , "dropcloth") 
		s=s.replace("earth grow" , "earthgro") 
		s=s.replace("eath grow" , "earthgro") 
		s=s.replace("earth-gro" , "earthgro") 
		s=s.replace("fantdoor" , "fandoor") 
		s=s.replace("fan door" , "fandoor") 
		s=s.replace("hex head" , "hexhead") 
		s=s.replace("hex-head" , "hexhead") 
		s=s.replace("hex pin" , "hexpin") 
		s=s.replace("hex-pin" , "hexpin") 
		s=s.replace("key stock" , "keysstock") 
		s=s.replace("key-stock" , "keystock") 
		s=s.replace("low voltage" , "lowvoltage") 
		s=s.replace("nail gun" , "nailgun") 
		s=s.replace("non adhesive" , "nonadhesive") 
		s=s.replace("non-adhesive" , "nonadhesive") 
		s=s.replace("claw foot","clawfoot") 
		s=s.replace("tow-behind" , "towbehind") 
		s=s.replace("tow behind" , "towbehind") 
		s=s.replace("inchespvc" , "inches pvc") 
		s=s.replace("inchesbrasswalltube" , "inches brass wall tube ") 
		s=s.replace("1npt" , "1 npt") 
		s=s.replace("frenchdoor","french door") 
		s=s.replace("horse power","horsepower") 
		s=s.replace("seriestruease"," series truease") 
		s=s.replace("tru-ease","truease") 
		s=s.replace("403esprit","403 esprit") 
		s=s.replace("bulbs2","bulbs 2") 
		s=s.replace("bulbsbulbs","bulbs") 
		s=s.replace("in.white","inches white") 
		s=s.replace("inanchor","inches anchor") 
		s=s.replace("inches.x","inches x") 
		s=s.replace("inchesbathroom","inches bathroom") 
		s=s.replace("inchesside","inches side") 
		s=s.replace("inchled","inches led") 
		s=s.replace("inchtrack","inches track") 
		s=s.replace("inchvinyl","inches vinyl") 
		s=s.replace("acondicionado" , "air conditioner") 
		s=s.replace("aircondiioner" , "air conditioner") 
		s=s.replace("aircondition" , "air conditioner") 
		s=s.replace("airconditioner" , "air conditioner") 
		s=s.replace("alltyp" , "all type") 
		s=s.replace("antivibration" , "anti vibration") 
		s=s.replace("aspectmetal" , "aspect metal") 
		s=s.replace("backbrace" , "back brace") 
		s=s.replace("bathtubdoor" , "bathtub door") 
		s=s.replace("beveragecooler" , "beverage cooler") 
		s=s.replace("bicyclestorage" , "bicycle storage") 
		s=s.replace("brownbrick" , "brown brick") 
		s=s.replace("brownswitch" , "brown switch") 
		s=s.replace("bushbutton" , "bush button") 
		s=s.replace("cabinetupper" , "cabinet upper") 
		s=s.replace("cablenail" , "cable nail") 
		s=s.replace("cablrail" , "cable rail") 
		s=s.replace("cadetcordless" , "cadet cordless") 
		s=s.replace("carpetshampoo" , "carpet shampoo") 
		s=s.replace("catchbag" , "catch bag") 
		s=s.replace("chaincome" , "chain come") 
		s=s.replace("chainlink" , "chain link") 
		s=s.replace("chopsaw" , "chop saw") 
		s=s.replace("coatracks" , "coat racks") 
		s=s.replace("concretetile" , "concrete tile") 
		s=s.replace("conditionerriding" , "conditioner riding") 
		s=s.replace("convectionoven" , "convection oven") 
		s=s.replace("cordclass" , "cord class") 
		s=s.replace("cordlessrotary" , "cordless rotary") 
		s=s.replace("cordyellow" , "cord yellow") 
		s=s.replace("cornershower" , "corner shower") 
		s=s.replace("counterdepth" , "counter depth") 
		s=s.replace("deckpaint" , "deck paint") 
		s=s.replace("diamondplate" , "diamond plate") 
		s=s.replace("discontinuedbrown" , "discontinued brown") 
		s=s.replace("doubletowel" , "double towel") 
		s=s.replace("drillimpact" , "drill impact") 
		s=s.replace("deckerbattery","decker battery") 
		s=s.replace("edgetape" , "edge tape") 
		s=s.replace("electrichot" , "electric hot") 
		s=s.replace("entrydoors" , "entry doors") 
		s=s.replace("exteriordoors" , "exterior doors") 
		s=s.replace("exteriorpaint" , "exterior paint") 
		s=s.replace("faucetskitchen" , "faucets kitchen") 
		s=s.replace("finethread" , "fine thread") 
		s=s.replace("fireplacewater" , "fireplace water") 
		s=s.replace("firetreat" , "fire treat") 
		s=s.replace("flatbrace" , "flat brace") 
		s=s.replace("flexlock" , "flex lock") 
		s=s.replace("floorcleaners" , "floor cleaners") 
		s=s.replace("footaluminum" , "foot aluminum") 
		s=s.replace("forgednails" , "forged nails") 
		s=s.replace("framelessmirror" , "frameless mirror") 
		s=s.replace("fuelpellets" , "fuel pellets") 
		s=s.replace("gaperaser" , "gap eraser") 
		s=s.replace("garagedoor" , "garage door") 
		s=s.replace("gasnstove" , "gas stove") 
		s=s.replace("granitecounter" , "granite counter") 
		s=s.replace("gratewhite" , "great white") 
		s=s.replace("hallodoor" , "hallo door") 
		s=s.replace("handtools" , "hand tools") 
		s=s.replace("headlag" , "head lag") 
		s=s.replace("heaterconditioner" , "heater conditioner") 	
		s=s.replace("hotwater" , "hot water") 
		s=s.replace("interiorwith" , "interior with") 
		s=s.replace("ironboard" , "iron board") 
		s=s.replace("jimmyproof" , "jimmy proof") 
		s=s.replace("joisthangers" , "joist hangers") 
		s=s.replace("keydoor" , "key door") 
		s=s.replace("kitchenfaucet" , "kitchen faucet") 
		s=s.replace("kitchenover" , "kitchen over") 
		s=s.replace("knobreplacement" , "knob replacement") 
		s=s.replace("latchbolt" , "latch bolt") 
		s=s.replace("lawnscarpenter" , "lawns carpenter") 
		s=s.replace("ledbulb" , "led bulb") 
		s=s.replace("lightbulb" , "light bulb") 
		s=s.replace("lightsavannah" , "light savannah") 
		s=s.replace("lightsensor" , "light sensor") 
		s=s.replace("linesplice" , "line splice") 
		s=s.replace("locationlight" , "location light") 
		s=s.replace("machinehardwood" , "machine hardwood") 
		s=s.replace("mediumpond" , "medium pond") 
		s=s.replace("metallicpaint" , "metallic paint") 
		s=s.replace("mmbolt" , "mm bolt") 
		s=s.replace("mmfitting" , "mm fitting") 
		s=s.replace("mmsaw" , "mm saw") 
		s=s.replace("mmwood" , "mm wood") 
		s=s.replace("mounteddrop" , "mounted drop") 
		s=s.replace("mountreading" , "mount reading") 
		s=s.replace("multipanel" , "multi panel") 
		s=s.replace("multitool" , "multi tool") 
		s=s.replace("nailshand" , "nails hand") 
		s=s.replace("needlenose" , "needle nose") 
		s=s.replace("nickelshower" , "nickel shower") 
		s=s.replace("oakmoss" , "oak moss") 
		s=s.replace("oilrubbed" , "oil rubbed") 
		s=s.replace("ouncesball" , "ounces ball") 
		s=s.replace("ouncesrust" , "ounces rust") 
		s=s.replace("outdoorlounge" , "outdoor lounge") 
		s=s.replace("outdoorstorage" , "outdoor storage") 
		s=s.replace("papertowels" , "paper towels") 
		s=s.replace("pivotangle" , "pivot angle") 
		s=s.replace("plasticbathroom" , "plastic bathroom") 
		s=s.replace("portercable" , "porter cable") 
		s=s.replace("postlantern" , "post lantern") 
		s=s.replace("quarteround" , "quarter round") 
		s=s.replace("quickfire" , "quick fire") 
		s=s.replace("rainbarrel" , "rain barrel") 
		s=s.replace("ranshower" , "rain shower") 
		s=s.replace("rebarbender" , "rebar bender") 
		s=s.replace("residentialsteel" , "residential steel") 
		s=s.replace("rodiron" , "rod iron") 
		s=s.replace("rodskit" , "rod kit") 
		s=s.replace("scraperreplacement" , "scraper replacement") 
		s=s.replace("sealparts" , "seal parts") 
		s=s.replace("seateat" , "sea test") 
		s=s.replace("semirigid" , "semi rigid") 
		s=s.replace("sensoroutdoor" , "sensor outdoor") 
		s=s.replace("seriesimpact" , "series impact") 
		s=s.replace("settingfence" , "setting fence") 
		s=s.replace("sheetmetal" , "sheet metal") 
		s=s.replace("shelflike" , "shelf like") 
		s=s.replace("showerstall" , "shower stall") 
		s=s.replace("showerstorage" , "shower storage") 
		s=s.replace("sideout" , "side out") 
		s=s.replace("sidepanels" , "side panels") 
		s=s.replace("skillsaw" , "skill saw") 
		s=s.replace("solidconcrete" , "solid concrete") 
		s=s.replace("spotffree" , "spot free") 
		s=s.replace("spraypaint" , "spray paint") 
		s=s.replace("stairtreads" , "stair treads") 
		s=s.replace("steamwasher" , "steam washer") 
		s=s.replace("straitline" , "strait line") 
		s=s.replace("surfacemount" , "surface mount") 
		s=s.replace("switchplate" , "switch plate") 
		s=s.replace("tapecase" , "tape case") 
		s=s.replace("tapeexhaust" , "tape exhaust") 
		s=s.replace("terminalsnylon" , "terminals nylon") 
		s=s.replace("tiertrolley" , "tier trolley") 
		s=s.replace("tilebath" , "tile bath") 
		s=s.replace("tilesaw" , "tile saw") 
		s=s.replace("topdown" , "top down") 
		s=s.replace("toprail" , "top rail") 
		s=s.replace("topsealer" , "top sealer") 
		s=s.replace("underdeck" , "under deck") 
		s=s.replace("underhood" , "under hood") 
		s=s.replace("underthe" , "under the") 
		s=s.replace("vaporbarrier" , "vapor barrier") 
		s=s.replace("ventenatural" , "vent natural") 
		s=s.replace("vinylcorner" , "vinyl corner") 
		s=s.replace("vinylcove" , "vinyl cover") 
		s=s.replace("wallhugger" , "wall hugger") 
		s=s.replace("walloven" , "wall oven") 
		s=s.replace("wallpanel" , "wall panel") 
		s=s.replace("wallprimer" , "wall primer") 
		s=s.replace("warmlight" , "warm light") 
		s=s.replace("washerparts" , "washer parts") 
		s=s.replace("waterheater" , "water heater") 
		s=s.replace("weedeaters" , "weed eaters") 
		s=s.replace("whirlpoolgas" , "whirlpool gas") 
		s=s.replace("whirlpoolstainless" , "whirlpool stainless") 
		s=s.replace("whitecomposite" , "white composite") 
		s=s.replace("whitesilicone" , "white silicone") 
		s=s.replace("widecellular" , "wide cellular") 
		s=s.replace("windowbalance" , "window balance") 
		s=s.replace("wiremesh" , "wire mesh") 
		s=s.replace("wirenuts" , "wire nuts") 
		s=s.replace("woodceramic" , "wood ceramic") 
		s=s.replace("woodflooring" , "wood flooring") 
		s=s.replace("woodsaddle" , "wood saddle") 
		s=s.replace("worksurface" , "work surface") 
		s=s.replace("yardwaste" , "yard waste") 
		s=s.replace("airfilter","air filter") 
		s=s.replace("airguard","air guard") 
		s=s.replace("airstone","air stone") 
		s=s.replace("backerboard","backer board") 
		s=s.replace("backplate","back plate") 
		s=s.replace("backplates","back plates") 
		s=s.replace("backpocket","back pocket") 
		s=s.replace("baseboarders","base boarders") 
		s=s.replace("baseplate","base plate") 
		s=s.replace("bathroomvanity","bathroom vanity") 
		s=s.replace("batteryfor","battery for") 
		s=s.replace("bentgrass","bent grass") 
		s=s.replace("boxspring","box spring") 
		s=s.replace("bullnose","bull nose") 
		s=s.replace("channellock","channel lock") 
		s=s.replace("collectionchrome","collection chrome") 
		s=s.replace("colorhouse","color house") 
		s=s.replace("cooktops","cook tops") 
		s=s.replace("dooroil","door oil") 
		s=s.replace("downlight","down light") 
		s=s.replace("downrod","down rod") 
		s=s.replace("downrods","down rods") 
		s=s.replace("easylight","easy light") 
		s=s.replace("epoxyshield","epoxy shield") 
		s=s.replace("faucethandle","faucet handle") 
		s=s.replace("firebowl","fire bowl") 
		s=s.replace("fireglass","fire glass") 
		s=s.replace("flagmakers","flag makers") 
		s=s.replace("flushmate","flush mate") 
		s=s.replace("flushmount","flush mount") 
		s=s.replace("foodsaver","food saver") 
		s=s.replace("gallondrywall","gallon drywall") 
		s=s.replace("hammerdrill","hammer drill") 
		s=s.replace("handleset","handle set") 
		s=s.replace("handscraped","hand scraped") 
		s=s.replace("handshower","hand shower") 
		s=s.replace("hightop","high top") 
		s=s.replace("lightwall","light wall") 
		s=s.replace("nailhead","nail head") 
		s=s.replace("naturaltone","natural tone") 
		s=s.replace("panelboard","panel board") 
		s=s.replace("pipeinsulation","pipe insulation") 
		s=s.replace("powerbuilt","power built") 
		s=s.replace("powertool","power tool") 
		s=s.replace("powerwasher","power washer") 
		s=s.replace("rainshower","rain shower") 
		s=s.replace("rainsuit","rain suit") 
		s=s.replace("rightheight","right height") 
		s=s.replace("sawtooth","saw tooth") 
		s=s.replace("scaleblaster","scale blaster") 
		s=s.replace("screwgun","screw gun") 
		s=s.replace("securemount","secure mount") 
		s=s.replace("snowblower","snow blower") 
		s=s.replace("snowblowers","snow blowers") 
		s=s.replace("softspring","soft spring") 
		s=s.replace("spreadstone","spread stone") 
		s=s.replace("stemcartridge","stem cartridge") 
		s=s.replace("thickrubber","thick rubber") 
		s=s.replace("threadlocker","thread locker") 
		s=s.replace("topmetal","top metal") 
		s=s.replace("topmount","top mount") 
		s=s.replace("touchless","touch less") 
		s=s.replace("touchscreen","touch screen") 
		s=s.replace("ultralight","ultra light") 
		s=s.replace("valvefor","valve for") 
		s=s.replace("wallbase","wall base") 
		s=s.replace("wallcovering","wall covering") 
		s=s.replace("wallcoverings","wall coverings") 
		s=s.replace("wallmount","wall mount") 
		s=s.replace("wallmounted","wall mounted") 
		s=s.replace("wallplate","wall plate") 
		s=s.replace("waterpump","water pump") 
		s=s.replace("waterseal","water seal") 
		s=s.replace("wattsolar","watt solar") 
		s=s.replace("weatherguard","weather guard") 
		s=s.replace("weatherhead","weather head") 
		s=s.replace("weathershield","weather shield") 
		s=s.replace("whiteled","white led") 
		s=s.replace("worklight","work light") 
		s=s.replace("yardguard","yard guard") 
        # some title edits
        s = s.replace("&quot;"," ")
        s = s.replace("è_"," ")
        s = s.replace("å¡"," ")
        s = s.replace("Û"," ")
        s = s.replace("åÊ"," ")
        s = s.replace("ÛÒ"," ")
        s = s.replace("Ûª"," ")
        s = s.replace("ÛÜ"," ")
        s = s.replace("Û÷"," ")
        s = s.replace("ÈÀ"," ")
        s = s.replace("ã¢"," ")
        # some title edits END
        s = s.replace("&#39;s"," ")
        #s = re.sub(r"[+-]\d+(:\.\d+)", lambda x: " " + p.number_to_words(x.group()) + " ", s)
        s = s.replace(" in."," 1in ")
        s = s.replace(" ft."," 1ft ")
        s = s.replace(" ft "," 1ft ")
        s = s.replace(" lb."," 1lb ")
        s = s.replace(" lb "," 1lb ")
        s = s.replace(" sq."," 1sq ")
        s = s.replace(" sq "," 1sq ")
        s = s.replace(" cu."," 1cu ")
        s = s.replace(" cu "," 1cu ")
        s = s.replace(" gal."," 1gal ")
        s = s.replace(" gal "," 1gal ")
        s = s.replace(" oz."," 1oz ")
        s = s.replace(" oz "," 1oz ")
        s = s.replace(" cm."," 1cm ")
        s = s.replace(" cm "," 1cm ")
        s = s.replace(" mm."," 1mm ")
        s = s.replace(" mm "," 1mm ")
        s = s.replace(" amp."," 1amp ")
        s = s.replace(" v. "," 1volt ")
        s = s.replace(" w. ", " 1watt ")
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.replace("U.S."," US ")
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("Ã¥Â¡"," ")
        s = s.replace("+"," plus ")
        s = s.replace(";"," ")
        s = s.replace(":"," ")
        s = s.replace("&amp;"," ")
        s = s.replace("&amp"," ")
        #s = s.replace(""," ")
        s = s.replace("-"," ")
        s = s.replace("#"," ")
        s = s.replace("("," ")
        s = s.replace(")"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ovr ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
        s = re.sub(r"(^\.|/)", r" ", s)
        s = re.sub(r"(\.|/)$", r" ", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x "," xbi ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*"," xbi ")
        s = s.replace(" by "," xbi ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.", r"\1in ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.", r"\1ft ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.", r"\1lb ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq)\.", r"\1sq ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu)\.", r"\1cu ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.", r"\1gal ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.", r"\1oz ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.", r"\1cm ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.", r"\1mm ", s)
        s = re.sub(r"([0-9]+)( *)(degrees|degree)\.", r"\1deg ", s)
        s = re.sub(r"([0-9]+)( *)(volts|volt)\.", r"\1volt ", s)
        s = re.sub(r"([0-9]+)( *)(watts|watt)\.", r"\1watt ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.", r"\1amp ", s)
        s = s.replace("&"," ")
        s = s.replace("'"," ")
        #s = s.replace("  "," ")
        s = s.replace(" . "," ")
        #s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        #s = (" ").join([z for z in s.split(" ")])
        #s = s.lower()
        #s = ' '.join(s.split())
        return s
    else:
        return "null"
        

print ("String clean ups...")        
df_all['new_search_term'] = df_all['new_search_term'].astype(str).map(lambda x:str_stem(x))#comment out corrector in str_stem function
df_all['product_title'] = df_all['product_title'].astype(str).map(lambda x:str_stem(x))
df_all['product_description'] = df_all['product_description'].astype(str).map(lambda x:str_stem(x))
df_all['brand'] = df_all['brand'].astype(str).map(lambda x:str_stem(x))
#df_all['brand']  = df_all['brand'].fillna('unbrand')
df_all['brand'] = np.where(df_all['brand'].isin ([' ','n a ','na',' na','nan','']),'unbrand',df_all['brand'])
df_all['material'] = df_all['material'].astype(str).map(lambda x:str_stem(x))
df_all['colour'] = df_all['colour'].astype(str).map(lambda x:str_stem(x)


df_all.to_csv(_path+'input/df_all_vect1.csv',index=False)

############################################################################################################################################


import time
start_time = time.time()

import numpy as np
import pandas as pd
import re

import sys
reload (sys)
sys.setdefaultencoding('ISO-8859-1')

import warnings
warnings.filterwarnings('ignore')

_path = '/home/rajdeep.banerjee/test/'


df_all = pd.read_csv(_path + 'input/df_all_vect1.csv',encoding = 'ISO-8859-1')
print df_all.shape


from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(min_df=5,  max_features=None, ngram_range=(1, 1))

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(algorithm='randomized',n_components=500, n_iter=5, random_state=None, tol=0.0)


tfv_srch_term = svd.fit_transform(tfv.fit_transform(df_all['new_search_term'].map(lambda x: re.sub('[^A-Za-z]',' ',x))))
tfv_pdt_ttl = svd.fit_transform(tfv.fit_transform(df_all['product_title'].map(lambda x: re.sub('[^A-Za-z]',' ',x))))
tfv_pdt_desc = svd.fit_transform(tfv.fit_transform(df_all['product_description'].map(lambda x: re.sub('[^A-Za-z]',' ',x))))

    
def seg_words(str1, str2):
    str2 = str2.lower()
    str2 = re.sub("[^a-z0-9./]"," ", str2)
    str2 = [z for z in set(str2.split()) if len(z)>2]
    words = str1.lower().split(" ")
    s = []
    for word in words:
        if len(word)>3:
            s1 = []
            s1 += segmentit(word,str2,True)
            if len(s)>1:
                s += [z for z in s1 if z not in ['er','ing','s','less'] and len(z)>1]
            else:
                s.append(word)
        else:
            s.append(word)
    return (" ".join(s))

def segmentit(s, txt_arr, t):
    st = s
    r = []
    for j in range(len(s)):
        for word in txt_arr:
            if word == s[:-j]:
                r.append(s[:-j])
                #print(s[:-j],s[len(s)-j:])
                s=s[len(s)-j:]
                r += segmentit(s, txt_arr, False)
    if t:
        i = len(("").join(r))
        if not i==len(st):
            r.append(st[i:])
    return r

def str_common_word(str1, str2):
    words, cnt = str1.split(), 0
    for word in words:
        if str2.find(word)>=0:
            cnt+=1
    return cnt

def str_whole_word(str1, str2, i_):
    cnt = 0
    while i_ < len(str2):
        i_ = str2.find(str1, i_)
        if i_ == -1:
            return cnt
        else:
            cnt += 1
            i_ += len(str1)
    return cnt
    
def get_n_gram_string_similarity(s1, s2, n):
    '''
    Helper function to get the n-gram "similarity" between two strings,
    where n-gram similarity is defined as the percentage of n-grams
    the two strings have in common out of all of the n-grams across the
    two strings.
    '''
    s1 = set(get_n_grams(s1, n))
    s2 = set(get_n_grams(s2, n))
    if len(s1.union(s2)) == 0:
        return 0
    else:
        return float(len(s1.intersection(s2)))/float(len(s1.union(s2)))

def get_n_grams(s, n):
    '''
    Helper function that takes in a string and the degree of n gram n and returns a list of all the
    n grams in the string. String is separated by space.
    '''
    token_pattern = re.compile(r"(?u)\b\w+\b")
    word_list = token_pattern.findall(s)
    n_grams = []
    if n > len(word_list):
        return []
    for i, word in enumerate(word_list):
        n_gram = word_list[i:i+n]
        if len(n_gram) == n:
            n_grams.append(tuple(n_gram))
    return n_grams


def extract_features(data):
    '''
    Perform feature extraction for variables that can be extracted
    the same way for both training and test data sets. The input
    "data" is the pandas dataframe for the training or test sets.
    '''
    token_pattern = re.compile(r"(?u)\b\w+\b")
    data["srch_tokens_in_title"] = 0.0
    data["srch_tokens_in_description"] = 0.0
    data["percent_srch_tokens_in_description"] = 0.0
    data["percent_srch_tokens_in_title"] = 0.0
    for i, row in data.iterrows():
        query = set(x.lower() for x in token_pattern.findall(row["new_search_term"]))
        title = set(x.lower() for x in token_pattern.findall(row["product_title"]))
        description = set(x.lower() for x in token_pattern.findall(row["product_description"]))
        if len(title) > 0:
            data.set_value(i, "srch_tokens_in_title", float(len(query.intersection(title)))/float(len(title)))
            data.set_value(i, "percent_srch_tokens_in_title", float(len(query.intersection(title)))/float(len(query)))
        if len(description) > 0:
            data.set_value(i, "srch_tokens_in_description", float(len(query.intersection(description)))/float(len(description)))
            data.set_value(i, "percent_srch_tokens_in_description", float(len(query.intersection(description)))/float(len(query)))
        #2 grams
        two_grams_in_query = set(get_n_grams(row["new_search_term"], 2))
        two_grams_in_title = set(get_n_grams(row["product_title"], 2))
        two_grams_in_description = set(get_n_grams(row["product_description"], 2))
        data.set_value(i, "two_grams_in_q_and_t", len(two_grams_in_query.intersection(two_grams_in_title)))
        data.set_value(i, "two_grams_in_q_and_d", len(two_grams_in_query.intersection(two_grams_in_description)))
        data.set_value(i, "two_grams_sim_in_q_and_t", get_n_gram_string_similarity(row["new_search_term"],row["product_title"],2))
        data.set_value(i, "two_grams_sim_in_q_and_d", get_n_gram_string_similarity(row["new_search_term"],row["product_description"],2))
        #3 grams
        three_grams_in_query = set(get_n_grams(row["new_search_term"], 3))
        three_grams_in_title = set(get_n_grams(row["product_title"], 3))
        three_grams_in_description = set(get_n_grams(row["product_description"], 2))
        data.set_value(i, "three_grams_in_q_and_t", len(three_grams_in_query.intersection(three_grams_in_title)))
        data.set_value(i, "three_grams_in_q_and_d", len(three_grams_in_query.intersection(three_grams_in_description)))
        data.set_value(i, "three_grams_sim_in_q_and_t", get_n_gram_string_similarity(row["new_search_term"],row["product_title"],3))
        data.set_value(i, "three_grams_sim_in_q_and_d", get_n_gram_string_similarity(row["new_search_term"],row["product_description"],3))


extract_features(df_all)

    
df_all['product_info'] = df_all['new_search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
#lengths
df_all['len_of_query'] = df_all['new_search_term'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)

df_all['brand'] = np.where(df_all['brand']=='','unknown',df_all['brand'])
df_all['brand']  = df_all['brand'].fillna('unbrand')
df_all['brand'] = np.where(df_all['brand'].isin ([' ','n a ','na',' n a',' na','nan','']),'unbrand',df_all['brand'])
df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)

df_all['material'] = np.where(df_all['material']=='','unknown',df_all['material'])
df_all['len_of_material'] = df_all['material'].map(lambda x:len(x.split())).astype(np.int64)

df_all['colour'] = np.where(df_all['colour']=='','unknown',df_all['colour'])
df_all['len_of_color'] = df_all['colour'].map(lambda x:len(x.split())).astype(np.int64)

df_all['new_search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))
df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))

#query last word match
df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))

#common words between search and product title/description
df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))
#df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
#df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']

#brand in searchterm
df_all['attr'] = df_all['new_search_term']+"\t"+df_all['brand']
df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
#brand in product title
df_all['attr'] = df_all['product_title']+"\t"+df_all['brand']
df_all['brand_in_title'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand_title'] = df_all['brand_in_title']/df_all['len_of_brand']
#brand in product desc
df_all['attr'] = df_all['product_description']+"\t"+df_all['brand']
df_all['brand_in_desc'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_brand_desc'] = df_all['brand_in_desc']/df_all['len_of_brand']

#material in searchterm
df_all['attr'] = df_all['new_search_term']+"\t"+df_all['material']
df_all['word_in_material'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_material_search']=df_all['word_in_material']/df_all['len_of_material']
#material in product title
df_all['attr'] = df_all['product_title']+"\t"+df_all['material']
df_all['material_in_title'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_material_title']=df_all['material_in_title']/df_all['len_of_material']
#material in product desc
df_all['attr'] = df_all['product_description']+"\t"+df_all['material']
df_all['material_in_desc'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_material_desc']=df_all['material_in_desc']/df_all['len_of_material']


#color in searchterm
df_all['attr'] = df_all['new_search_term']+"\t"+df_all['colour']
df_all['word_in_color'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_color_search']=df_all['word_in_color']/df_all['len_of_color']
#color in product title
df_all['attr'] = df_all['product_title']+"\t"+df_all['colour']
df_all['color_in_title'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_color_title']=df_all['color_in_title']/df_all['len_of_color']
#color in product desc
df_all['attr'] = df_all['product_description']+"\t"+df_all['colour']
df_all['color_in_desc'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
df_all['ratio_color_desc']=df_all['color_in_desc']/df_all['len_of_color']



df_attr = pd.read_csv(_path+'input/attributes.csv',encoding="ISO-8859-1")
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
df_material = df_attr[df_attr.name == "Material"][["product_uid", "value"]].rename(columns={"value": "material"})
df_brand = pd.unique(df_all.brand.ravel())

d={}
i = 1000
for s in df_brand:
    d[s]=i
    i+=3
    
df_material = pd.unique(df_all.material.ravel())
dmat={}
i = 100000
for s in df_material:
    dmat[s]=i
    i+=3
    

df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
df_all['material_feature'] = df_all['material'].map(lambda x:dmat[x])

df_all['search_term_feature'] = df_all['new_search_term'].map(lambda x:len(x))


import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics.pairwise import cosine_similarity
#dst = cosine_similarity(tfv_srch_term,tfv_pdt_ttl).diagonal()

dst_srch_ttl = np.zeros(df_all.shape[0])
for i in range(df_all.shape[0]):
    d1 = tfv_srch_term[i,:]
    d2 = tfv_pdt_ttl[i,:]
    dst_srch_ttl[i] = cosine_similarity(d1,d2)
    
dst_srch_desc = np.zeros(df_all.shape[0])
for i in range(df_all.shape[0]):
    d1 = tfv_srch_term[i,:]
    d2 = tfv_pdt_desc[i,:]
    dst_srch_desc[i] = cosine_similarity(d1,d2)

df_all['srch_title_cos_sim']=list(dst_srch_ttl)
df_all['srch_desc_cos_sim']=list(dst_srch_desc)


#Jaccard distance 
from __future__ import division
   
def jaccard_similarity(query,doc):
    intersection = set(query).intersection(set(doc))
    union = set(query).union(set(doc))
    return len(intersection)/len(union)

tokenize = lambda doc : doc.lower().split(' ')

def func_ttl(row):
    return jaccard_similarity(tokenize(row['new_search_term']),tokenize(row['product_title']))

df_all['srch_title_jaccard_sim'] = df_all.apply(func_ttl,axis=1)

def func_desc(row):
    return jaccard_similarity(tokenize(row['new_search_term']),tokenize(row['product_description']))

df_all['srch_desc_jaccard_sim'] = df_all.apply(func_desc,axis=1)

df_all.to_csv(_path+'input/df_all_new_feat2.csv',index=False)


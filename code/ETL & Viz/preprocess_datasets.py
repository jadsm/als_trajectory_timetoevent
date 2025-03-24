# preprocess_datasets.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file preprocesses datasets


import pandas as pd
import numpy as np
import sys
sys.path.append('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/best_models')
from utils.utils import *

reload_popt = False
reload_class_features = False

# set up
path = '/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv'

# load feature data
df_master = pd.read_csv(path,encoding='latin_1',low_memory=False)
df_static = pd.read_csv('data/static_22_03_24.csv')
df_phenotype_mapper = pd.read_csv('data/phenotype_mapper.csv')

# add static features
static_cols = [k for k in df_static.keys() if not k in df_master.keys()]
df_master = df_master.merge(df_static.loc[:,['id']+static_cols].rename(columns={'id':'id_old'}),on='id_old',how='left')

# features km
features_km(df_master)

# get initial features
df_features,dfcens = get_initial_features(df_master,df_phenotype_mapper)

# classes data
# df_classes = pd.read_csv('data/Training_classes.csv') # this is for training classes only

# get the class features
df_class_feats = get_class_features_gt()
df_class_feats2 = get_class_features_frechet(df_class_feats,df_features,reload_class_features=reload_class_features)
# export classes for evaluation
df_class_feats.merge(df_class_feats2,on='id',suffixes=['_gt','_est']).to_csv('data/classfeatures_frechet_and_gt.csv',index=False)

# reprocess raw features
df_features0 = df_features.copy()
X,y,ids,df_features = get_raw_feartures(df_features,dfcens)

# get decay features
dfdecay,y = get_decay_features(dfcens,df_features,ids,reload_popt=reload_popt)

# define encals features
encals_cols = ['sex_Female','site_onset_Spinal', 'age_at_onset', 'diagnostic_delay_months','ALSFRS_Slope_Onset_to_FirstALSFRS','mean_fev','C9orf72']

# merge all the features
decay_cols = [k for k in dfdecay.keys() if k not in df_features.keys()]
raw_cols = [k for k in df_features.keys() if k not in list(dfdecay.keys())+encals_cols]
demo_cols = [k for k in df_features.keys() if k in list(dfdecay.keys())]
df_allfeatures = df_class_feats.merge(pd.concat([dfcens.loc[:,'id'],dfdecay.loc[:,decay_cols]],axis=1),on='id',
                                      how='outer').merge(pd.concat([dfcens.loc[:,'id'],df_features],axis=1),
                                                         on='id',how='outer')

# add FEV for encals features
df_allfeatures['mean_fev'] = df_allfeatures.loc[:,'d0_% predicted':'d1080_% predicted'].mean(axis=1)
# df_encals = df_features.loc[:,encals_cols]
df_allfeatures.to_csv('data/allfeatures.csv',index=False)

dfcens.to_csv('data/cens_death.csv',index=False)
dfcens.drop(columns=['Dead','Death_Date'],inplace=True)
dfcens.to_csv('data/cens.csv',index=False)

# calculate the feature keys
featurekey = pd.DataFrame(raw_cols,columns=['feature'])
featurekey['type'] = 'raw'
featurekey2 = pd.DataFrame(decay_cols,columns=['feature'])
featurekey2['type'] = 'decay'
featurekey3 = pd.DataFrame(df_class_feats.iloc[:,1:].keys(),columns=['feature'])
featurekey3['type'] = 'class'
featurekey4 = pd.DataFrame(demo_cols,columns=['feature'])
featurekey4['type'] = 'demo'
featurekey5 = pd.DataFrame(encals_cols,columns=['feature'])
featurekey5['type'] = 'encals'
featurekey = pd.concat([featurekey,featurekey2,featurekey3,featurekey4,featurekey5],axis=0,ignore_index=True)
featurekey.to_csv('data/featurekey.csv',index=False)
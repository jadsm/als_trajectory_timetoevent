# train_right_censored.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file trains the model and provides exhaustive results.

import pandas as pd
import os
from utils.utils_censor import *
import logging
from datetime import datetime
from multiprocessing import Pool

reload = True

# Basic configuration
logging.basicConfig(filename=f'logs/Surv_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pred.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    models_df = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/time_prediction/data/results_strat_xgbmaepo_weighted_all6_no_imputation.csv')
    # models_df.groupby(['test_fold','method','dataset'])['Cindex_test_uncensored'].describe()
    models_df = models_df.query('is_best_trial == True and dataset in ("demo", "classes_neqm")')
    # combine both ,"demo+classes_neqm"
    # aux = models_df.query('dataset == "classes_neqm"')
    # aux['dataset'] = "demo+classes_neqm"
    # models_df = pd.concat([models_df,aux],axis=0,ignore_index=True)

    # create keys 
    models_df['key'] = models_df['dataset']+models_df['method']
    if not reload:
        computed_models = [pd.read_csv(os.path.join('logs',l)) for l in os.listdir('logs') if l.endswith('.csv') and l.startswith('simul')]
        if len(computed_models)>0:
            computed_models = pd.concat(computed_models)
            computed_models['key'] = computed_models['dataset']+computed_models['method']
            models_df = models_df.query(f"key not in {tuple(computed_models['key'].unique())}").reset_index(drop=True)
        else:
            print('reloading all...')
    # models_df = models_df.loc[:2,:]

    # load data
    fixed_args = load_data() + tuple([logger])
    input_data = (tuple([row])+fixed_args for ri,row in models_df.iterrows())

    if paralell:
        # compute all data
        with Pool(14) as p:     
            # input_data = list(product(datasets, methods))
            res = p.starmap(train_all_methods_wrapper, input_data)
    else:
        res = []
        for pars in input_data:
            res.append(train_all_methods_wrapper(*pars))
    
    # decode the results
    results,summary,dfpred = [],[],[]
    for r in res:
        summary.append(r[1])
        results.append(r[0])
        dfpred.append(r[2])
    results = pd.concat(results,axis=0,ignore_index=True)
    summary = pd.concat(summary,axis=0,ignore_index=True)
    dfpred = pd.concat(dfpred,axis=0,ignore_index=True)
    results.to_csv('data/acc_curve_simulation_last0_5Final_newclass63mice.csv',index=False)
    summary.to_csv('data/summary_database_rotation_best_last0_5Final_newclass63mice.csv',index=False)
    dfpred.to_csv('data/predictions_last0_5Final_newclass63mice.csv',index=False)

    # summary_sum = summary.loc[:,['method',	'dataset',
    #                'PredIn90_test_uncensored','PredIn180_test_uncensored','PredIn360_test_uncensored',
    #                'MedianAE_test_uncensored','Cindex_test_uncensored']]
    # summary_sum = summary_sum.groupby(['method','dataset']).mean().reset_index()
    
    # # minmax = MinMaxScaler()
    # # summary_sum['overall_score'] = 1/3*(summary_sum['Acc90_test_uncensored'].values.reshape(-1,1)+summary_sum['Cindex_test_uncensored'].values.reshape(-1,1)+1-minmax.fit_transform(summary_sum['MedianAE_test_uncensored'].values.reshape(-1,1)))
    # # summary_sum.sort_values(by=['overall_score'],ascending=False,inplace=True)
    # summary_sum.to_csv('data/summary_database_rotation_best_last0_5Final_avg_newclass62.csv',index=False)

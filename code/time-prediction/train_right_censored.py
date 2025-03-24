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
    # load the pre-hyperpameter opt data
    models_df = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/best_models/data/results_strat_xgbmaepo_weighted_all3Final.csv').query('is_best_trial == True')
    # filter if necessary
    # models_df = models_df.query('method == "SkSurvCoxLinear" and dataset in ("classes+decay","classes")').reset_index(drop=True)
    
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
    
    # load data
    fixed_args = load_data() + tuple([logger])
    input_data = (tuple([row])+fixed_args for ri,row in models_df.iterrows())

    if paralell:
        # compute all data
        with Pool(6) as p:     
            # input_data = list(product(datasets, methods))
            res = p.starmap(train_all_methods, input_data)
    else:
        res = []
        for pars in input_data:
            res.append(train_all_methods(*pars))
    
    # decode the results
    results,summary,dfpred = [],[],[]
    for r in res:
        summary.append(r[1])
        results.append(r[0])
        dfpred.append(r[2])
    results = pd.concat(results,axis=0,ignore_index=True)
    summary = pd.concat(summary,axis=0,ignore_index=True)
    dfpred = pd.concat(dfpred,axis=0,ignore_index=True)
    # sink data
    results.to_csv('data/acc_curve_simulation_last0_5Final.csv',index=False)
    summary.to_csv('data/summary_database_rotation_best_last0_5Final.csv',index=False)
    dfpred.to_csv('data/predictions_last0_5Final.csv',index=False)

    summary_sum = summary.loc[:,['method',	'dataset',
                   'PredIn90_test_uncensored','PredIn180_test_uncensored','PredIn360_test_uncensored',
                   'MedianAE_test_uncensored','Cindex_test_uncensored']]
    summary_sum = summary_sum.groupby(['method','dataset']).mean().reset_index()
    
    summary_sum.to_csv('data/summary_database_rotation_best_last0_5Final_avg.csv',index=False)

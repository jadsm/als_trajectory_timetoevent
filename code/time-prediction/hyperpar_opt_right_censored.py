# hyperpar_opt_right_censored.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file allows to find best hyperparameters

from multiprocessing import Pool
from itertools import product
from datetime import datetime
from utils.utils_censor import *
import logging
import os

# Basic configuration
logging.basicConfig(filename=f'logs/Surv_{datetime.now().strftime("%Y%m%d_%H%M%S")}_train.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    computed_methods = [] if ignore_computed_methods else [(l.split('_')[0],l.split('_')[1]) for l in os.listdir('logs') if l.endswith('.csv')] 

    # set combinations of models
    fixed_args = load_data() + (logger,seed,n_trials)
    input_data = ((d, m)+fixed_args for d, m in product(datasets,methods) if not (d,m) in computed_methods)

    if paralell:
        # compute all data
        with Pool(6) as p:
            results = p.starmap(hyperopt_all_methods, input_data)
    else:
        results = []
        for pars in list(input_data):
            print(pars)
            results.append(hyperopt_all_methods(*pars))

    # export all
    results = pd.concat(results,axis=0)
    results['is_best_trial'] = results['best_trial']==results['trial']
    results.to_csv('data/results_strat_xgbmaepo_weighted_all6_no_imputation.csv',index=False)
    # results.to_csv('data/results_all_SurvFinal.csv',index=False)
    
    # get a summary
    # results = pd.read_csv('data/results_all_SurvFinal.csv')
    best_results = results.query('is_best_trial== True').sort_values(by=['Cindex_test_uncensored'],ascending=False).drop_duplicates(subset=['method','dataset'],keep='first')
    # best_results.to_csv('data/results_best_SurvFinal.csv',index=False)
    best_results.to_csv('data/results_best_strat_xgbmaepo_weighted_all6_no_imputation.csv',index=False)

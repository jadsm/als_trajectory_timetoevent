# death_model_eval.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file evaluates the performance of death models.


import pandas as pd
import numpy as np
from utils.utils import cal_metrics,infer_new_ALSFRS_slope,load_death_features
from utils.utils_censor import SkSurvCoxLinear, calc_all_metrics,XGBoostMAEPOReg,get_df_pred,calculate_perc_survival_time
import pickle

X, y,dffeat,df_unc = load_death_features('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv',
                                  'data/encals_overall_survival_pred_final.csv')

out = pd.DataFrame()
dfpred = pd.DataFrame()
res_encals_death = []
for test_rotation in ['IDPP','ArQ','PROACT']:
    ###### evaluation encals model
    aux = df_unc.query(f'Database == "{test_rotation}"')
    metrics_encals = cal_metrics(aux['Death_Date']*365.25/12,
                                 aux['OUT']*365.25/12)
    metrics_encals['n'] = aux.shape[0]
    metrics_encals['test_rotation'] = test_rotation
    metrics_encals['method'] = 'Death_encals'
    res_encals_death.append(metrics_encals)
    
    ##### fit our own model
    train_valid_idx,test_idx = dffeat.query(f'Database != "{test_rotation}"').index,dffeat.query(f'Database == "{test_rotation}"').index

    clf2 = SkSurvCoxLinear()
    dtrain = clf2.dmat_builder(X[train_valid_idx],y[train_valid_idx])
    dtest = clf2.dmat_builder(X[test_idx],y[test_idx])
    y_pred,y_pred_train = clf2.compute_test_pred(None, dtrain, dtest,params_df={'alpha':0.07})
    out1 = calc_all_metrics(pd.DataFrame(),None,clf2,0,1300,X,y,y_pred,y_pred_train,train_valid_idx,test_idx)
    out1['method'] = 'LinearCOXPH'
    out1['test_rotation'] = test_rotation
    out1['n'] = y_pred.shape[0]
    clf2.predict_baseline_hazard(pd.DataFrame(y[train_valid_idx]))
    # calculate the rest
    y_pred_test = calculate_perc_survival_time(clf2,X[test_idx],y[test_idx],f=.5)
    y_pred_train = calculate_perc_survival_time(clf2,X[train_valid_idx],y[train_valid_idx],f=.5)

    dfpred1 = get_df_pred(y[train_valid_idx],y[test_idx],y_pred_train,y_pred_test,train_valid_idx,test_idx,test_rotation,'LinearCOXPH','ENCALS_death')
    
    clf3 = XGBoostMAEPOReg()
    dtrain = clf3.dmat_builder(X[train_valid_idx],y[train_valid_idx])
    dtest = clf3.dmat_builder(X[test_idx],y[test_idx])
    y_pred,y_pred_train = clf3.compute_test_pred(None, dtrain, dtest,params_df={'learning_rate':0.08,'max_depth': 8,'reg_alpha': 30,'reg_lambda': 5})

    out2 = calc_all_metrics(pd.DataFrame(),None,clf3,0,1300,X,y,y_pred,y_pred_train,train_valid_idx,test_idx)
    out2['method'] = 'XGBoostReg'
    out2['test_rotation'] = test_rotation
    out2['n'] = y_pred.shape[0]
    dfpred2 = get_df_pred(y[train_valid_idx],y[test_idx],y_pred_train,y_pred,train_valid_idx,test_idx,test_rotation,'XGBoostReg','ENCALS_death')
    
    out = pd.concat([out,out1,out2],axis=0)
    # out.index = ['LinearCOXPH','XGBoostReg']
    
    dfpred = pd.concat([dfpred,dfpred1,dfpred2],axis=0)
    
    with open(f'models/death_XGBoostReg_{test_rotation}.pkl', 'wb') as fid:
        pickle.dump(clf3, fid)

cols = ['MSE_test_uncensored','RMSE_test_uncensored','R2_test_uncensored','MAE_test_uncensored','MedianAE_test_uncensored','PredIn90_test_uncensored','PredIn180_test_uncensored','PredIn360_test_uncensored','Cindex_test_uncensored',
        'method','test_rotation','n']
out = out.loc[:,cols]
out.columns = [c.replace('_test_uncensored','') for c in out.columns]

# merge all
out2 = pd.DataFrame(res_encals_death)
out = pd.concat([out,out2],axis=0)
out['dataset'] = 'ENCALS_death'
out.to_csv('data/death_prediction_performance.csv',index=False)

dfpred.to_csv('data/death_predictions.csv',index=False)

# utils_censor.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# These are utility classes and functions for right censoring modelling
# This file contains base code developed here: https://doi.org/10.48550/arXiv.2006.04920

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import time
import altair as alt
import pandas as pd
from sksurv.util import Surv
from sksurv.metrics import concordance_index_ipcw
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.model_selection import StratifiedKFold
from scipy.optimize import curve_fit
import optuna
from optuna.samplers import RandomSampler
from datetime import datetime
from utils.utils import cal_metrics,cal_predintol,median_absolute_error
import time
from uuid import uuid4
import pickle
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import copy
from typing import Tuple
from sksurv.nonparametric import kaplan_meier_estimator
from scipy.interpolate import interp1d
from utils.constants import *


# datasets = ['encals','classes', 'decayfeatures','classes+decay']# ,'rawfeatures'
datasets = ['demo','classes_neqm']
# datasets = ['demo','classes','classes_eq','classes_neq','classes_slope','classes_neqm']
# datasets = ['classes','classes_eq','classes_neq']
methods = ['SkSurvCoxLinear','XGBoostCox','XGBoostMAEPOReg']
# methods = ['XGBoostCox','SkSurvCoxLinear','XGBoostMAEPOReg']#,'SVRAFT' 'SkSurvCoxLinear','SVRAFT','RFAFT', '

mice_imputation = False
test_only = False
seed,n_trials = 1,5
bootstrap_flag = True
paralell = True
ignore_computed_methods = True
f_threshold = .5
hazard_ratio_scale = .9    
n_iterations = 100
bootstrap_frac = .8

def remove_nans(X00,y00,strat_var0,col_threshold = 4000):
    cols_to_keep = np.isnan(X00.astype(float)).sum(axis=0)<col_threshold
    X = X00[:,cols_to_keep]
    idx_to_keep = np.invert(np.isnan(X).any(axis=1))
    X = X[idx_to_keep,:]
    event,time = zip(*y00)
    y = Surv.from_arrays(event=event, time=time)   
    y = y[idx_to_keep]
    strat_var = strat_var0[idx_to_keep]
    return X,y,strat_var,cols_to_keep

def get_df_pred(y_train,y_test,y_pred_train,y_pred_test,train_idx,test_idx,database_rotation,method,dataset):
    idx_unc_train = np.where([ii[0]==1 for ii in y_train])[0]
    idx_unc_test = np.where([ii[0]==1 for ii in y_test])[0]
    y_train_time = np.array([ii[1] for ii in y_train if ii[0]==1])
    y_test_time = np.array([ii[1] for ii in y_test if ii[0]==1])

    D = []
    for k in [[y_pred_test[idx_unc_test],test_idx[idx_unc_test],'pred','test'],[y_pred_train[idx_unc_train],train_idx[idx_unc_train],'pred','train'],
                [y_train_time,train_idx[idx_unc_train],'gt','train'],[y_test_time,test_idx[idx_unc_test],'gt','test']]:
        dfpred = pd.DataFrame(k[0],columns=[f'time_{k[2]}'])
        dfpred['pseudoid'] = k[1]
        dfpred['data_split'] = k[3]
        # dfpred['data_origin'] = k[2]
        D.append(dfpred.copy())
    dfpred1 = pd.concat(D[:2],axis=0)
    dfpred2 = pd.concat(D[2:],axis=0)
    dfpred = dfpred1.merge(dfpred2,on=['pseudoid', 'data_split'])
    dfpred['test_rotation'] = database_rotation
    dfpred['method'] = method
    dfpred['dataset'] = dataset
    print('shape dfpred:',dfpred.shape)
    return dfpred

def calc_all_metrics(out,study,model_obj,test_fold_id,dtrain_valid_combined,dtest,uncensored):
        
    train_uncensored,test_uncensored = uncensored

    # set up variables
    y_train = dtrain_valid_combined.get_label() if not isinstance(dtrain_valid_combined,tuple) else np.array([yy[1] for yy in dtrain_valid_combined[1]])
    y_test = dtest.get_label() if not isinstance(dtest,tuple) else np.array([yy[1] for yy in dtest[1]])

    if study is not None:
        aux = pd.DataFrame([s.params for s in study.trials],
                    index = range(len(study.trials))).reset_index(drop=False).rename(columns={'index':'trial'})
        params_df = None
    else:
        aux = pd.DataFrame([0],columns=['trial']) # create a dummy row
        params_df = model_obj.params_df
    y_train_uncensored = y_train[train_uncensored]
    y_test_uncensored = y_test[test_uncensored]
   
    for n_trial in aux['trial'].values:
        
        _,y_pred_train = model_obj.compute_test_pred(
                                    study=study, dtrain_valid_combined=dtrain_valid_combined, dtest=dtest,params_df=params_df,trial_num = n_trial)

        if model_obj.prediction_type == 'survival':
            # get the baseline hazard - this will be needed later in the lower level api implementations
            y_comb_train = pd.DataFrame([train_uncensored,y_train],index=['event','time']).T
            model_obj.predict_baseline_hazard(y_comb_train)
            X_test = dtest.get_data() if not isinstance(dtest,tuple) else dtest[0]
            X_train = dtrain_valid_combined.get_data() if not isinstance(dtrain_valid_combined,tuple) else dtrain_valid_combined[0]
            ynow = Surv.from_arrays(event=np.ones_like(y_test_uncensored).astype(bool),time=y_test_uncensored) if model_obj.method_name == "XGBoostCox" else y_test_uncensored
            y_pred_test_uncensored = calculate_perc_survival_time(model_obj,X_test[test_uncensored],ynow,f=f_threshold)
            ynow = Surv.from_arrays(event=np.ones_like(y_train_uncensored).astype(bool),time=y_train_uncensored) if model_obj.method_name == "XGBoostCox" else y_train_uncensored
            y_pred_train_uncensored = calculate_perc_survival_time(model_obj,X_train[train_uncensored],ynow,f=f_threshold)
            ynow = Surv.from_arrays(event=test_uncensored,time=y_test) if model_obj.method_name == "XGBoostCox" else y_test
            y_pred_test = calculate_perc_survival_time(model_obj,X_test,ynow,f=f_threshold)
            ynow = Surv.from_arrays(event=train_uncensored,time=y_train) if model_obj.method_name == "XGBoostCox" else y_train
            y_pred_train = calculate_perc_survival_time(model_obj,X_train,ynow,f=f_threshold)
        elif model_obj.prediction_type == 'time':
            y_comb_train = model_obj.dmat_builder(dtrain_valid_combined.get_data()[train_uncensored],
                                                  Surv.from_arrays(event=np.ones_like(y_train_uncensored).astype(bool),time=dtrain_valid_combined.get_label()[train_uncensored]))
            y_comb_test = model_obj.dmat_builder(dtest.get_data()[test_uncensored],
                                                 Surv.from_arrays(event=np.ones_like(y_test_uncensored).astype(bool),time=dtest.get_label()[test_uncensored]))
            y_pred_test_uncensored = model_obj.modelnow.predict(y_comb_test)
            y_pred_train_uncensored = model_obj.modelnow.predict(y_comb_train)
            y_pred_train = model_obj.modelnow.predict(dtrain_valid_combined)
            y_pred_test = model_obj.modelnow.predict(dtest)
        else:
            raise ValueError('prediction_type should be either time or survival!')

        metrics_train = cal_metrics(y_train,y_pred_train)
        metrics_test = cal_metrics(y_test,y_pred_test)
        metrics_train_uncensored = cal_metrics(y_train_uncensored,y_pred_train_uncensored)
        metrics_test_uncensored = cal_metrics(y_test_uncensored,y_pred_test_uncensored)

        for tr,tt,tru,ttu in zip(metrics_train.items(),metrics_test.items(),metrics_train_uncensored.items(),metrics_test_uncensored.items()):
            aux.loc[n_trial,tr[0]+'_train'] = tr[1]
            aux.loc[n_trial,tr[0]+'_test'] = tt[1]
            aux.loc[n_trial,tr[0]+'_train_uncensored'] = tru[1]
            aux.loc[n_trial,tr[0]+'_test_uncensored'] = ttu[1]
        # except Exception as e: 
        #     print(e)

    aux['test_fold'] = test_fold_id
    # aux['Cindex'] = cindex
    aux['best_trial'] = study.best_trial.number if study is not None else None
    # aux['brier'] = brier
    aux['best_value'] = study.best_value if study is not None else None
    return pd.concat([out,aux],axis=0)

def get_hazard_from_cum_hazard(Cum_Ho):
    if not isinstance(Cum_Ho,pd.DataFrame):
        Cum_Ho = pd.DataFrame(Cum_Ho.y,index=Cum_Ho.x,columns=['KM_estimate'])
        Cum_Ho.index.name = 'timeline'
    Cum_Ho_d = Cum_Ho.diff().reset_index()
    return pd.DataFrame(Cum_Ho_d['KM_estimate']/Cum_Ho_d['timeline'],index=Cum_Ho_d['timeline'],columns=['baseline_hazard'])
    
def predict_baseline_cum_hazard_generic(df):
    # df - is normally the training+validation data
    # df has time and event columns
    # note event is the reciprocal of cens - # df['event'] = ~df['cens']
    # it uses KM unbiased estimator but can use splines / RP splines / breslow
    # df.sort_values(by=['time'],inplace=True)
        
    # get the baseline hazard Ho(t)
    ################ NEEDS VALIDATION!! np.abs() is added for Cox model whose censoring is indicated by negative values
    km_time, km_survival_function, _ = kaplan_meier_estimator(df['event'].astype(bool).values, np.abs(df['time'].values), conf_type="log-log")
    Cum_Ho = -np.log(km_survival_function)
    Cum_Ho = pd.DataFrame(Cum_Ho,index=km_time)
    # Cum_Ho = Cum_Ho.reset_index()
    return Cum_Ho
    # Cum_Ho_d = Cum_Ho.diff()
    # Ho = pd.DataFrame(Cum_Ho_d['KM_estimate']/Cum_Ho_d['timeline'],index=Cum_Ho['timeline'],columns=['baseline_hazard'])
    # return get_hazard_from_cum_hazard(Cum_Ho)

def predict_survival_function_generic(df,Cum_Ho):
    # df is typically the testing or validation tests 
    # it contains a column with the predictions which are transformed into Hazard's ratio before plugging 
    # it into the Hazard estimation equation
    # Ho is the baseline hazard
    Cum_H = pd.concat([Cum_Ho*np.exp(p*hazard_ratio_scale) for p in df],axis=1)
    # Cum_H = H.cumsum()
    # survival function for each patient
    Surv = np.exp(-Cum_H)
    # Surv.columns = df.index
    Surv.columns = np.arange(1,len(df)+1)
    return Surv
    # gastrostomy probability
    # P = 1 - Surv
    # return P

def calculate_perc_survival_time(model_obj,X,y,f=.5):
    # get the matrix
    d = model_obj.dmat_builder(X, y)
    # get the survival function    
    survival_function = model_obj.predict_survival_function(d)

    aa = []
    for col in (survival_function):
        idx = survival_function.loc[survival_function[col]<f,col]
        if len(idx)==0:
            # aa.append(survival_function.index[-1])
            aa.append(-1)
        else:
            aa.append(idx.index[0])
    return np.array(aa)

def calc_bootstrap(model_obj,dfsummary,X_test,y_test,dtrain_valid_combined,train_uncensored,test_uncensored,study=None,test_fold_id=None):
    test_uncensored0 = test_uncensored.copy()
    start = time.time()
    n_size = int(len(X_test) * bootstrap_frac)
    A = []
    suffix = '_test_uncensored'
    for i in range(n_iterations):
        x = np.arange(X_test.shape[0])
        np.random.shuffle(x)
        # test_uncensored = np.ones(n_size).astype(bool)
        test_uncensored = test_uncensored0[x[:n_size]]
        dtest = model_obj.dmat_builder(X_test[x[:n_size],:], y_test[x[:n_size]])
        
        # y_pred_now,y_pred_train_now = model_obj.compute_test_pred(None, dtrain_valid_combined, dtest,params_df=model_obj.params_df)

        aux = calc_all_metrics(pd.DataFrame(),study,model_obj,test_fold_id,dtrain_valid_combined, dtest,[train_uncensored,test_uncensored])
        # aux = calc_all_metrics(pd.DataFrame(),None,model_obj,None,X_test,y_test,y_pred_train_now,x[:n_size],x[:n_size])

        A.append(aux.loc[:,['PredIn90'+suffix,'MedianAE'+suffix,'Cindex'+suffix]])
    
    A = pd.concat(A)
    
    for var,max in {'Cindex':1,'MedianAE':10000,'PredIn90':1}.items():
        dfsummary[f'{var}_mean'] = A.loc[:,var+suffix].mean()
        dfsummary[f'{var}_lb'] = np.clip(dfsummary[f'{var}_mean']-A.loc[:,var+suffix].std()*1.96,a_min=0,a_max=None)
        dfsummary[f'{var}_ub'] = np.clip(dfsummary[f'{var}_mean']+A.loc[:,var+suffix].std()*1.96,a_min=None,a_max=max)
    end = time.time()
    time_taken = end - start
    print(f'Time elapsed (boostrapping n={n_iterations}) = {time_taken}')
    dfsummary['elapsed_time_bootstrap'] = time_taken
    dfsummary['n_bootstraps'] = n_iterations
    dfsummary['bootstrap_frac'] = bootstrap_frac
    return dfsummary

def load_data(suffix=''):
    y0 = pd.read_csv('data/cens.csv')
    y0 = y0.loc[y0['tcens']>=0,:]
    
    # y0['tcens'] = np.clip(y0['tcens'],a_min=0,a_max=1080)
    # y0 = y0.loc[y0['tcens']<6000,:]
    feature_key = pd.read_csv('data/featurekey.csv')
    X0 = pd.read_csv(f'data/allfeatures{suffix}.csv')
    X0 = X0.merge(y0['id'],on='id',how='right').drop(columns='id')
    # y0['event'] = y0['cens'] == 0
    y00 = Surv.from_arrays(event=y0['event'].values, time=y0['tcens'].values)   
    tau = np.percentile(y0['tcens'].values[y0['event'].values], q=80)
    # stratification variable
    try:
        strat_var0 = X0.loc[:,'Database']
    except:
        strat_var0 = pd.concat([X0.loc[:,k].replace(True,k.split('_')[1]).replace(False,np.nan) for k in X0.keys() if k.startswith('Database')],axis=1)
        strat_var0['Database'] = strat_var0.iloc[:,0] 
        for k in strat_var0.keys()[1:]:
            strat_var0['Database'] = strat_var0['Database'].fillna(strat_var0[k])
        strat_var0 = strat_var0.drop(columns=strat_var0.keys()[:-1]).values

    return X0, y00, tau, strat_var0, feature_key

def xgb_train(*, trial, train_valid_folds, model_obj, valid_metric_func):
    params = model_obj.get_params(trial)
    params.update(model_obj.get_base_params())

    bst = []  # bst[i]: XGBoost model fit using i-th CV fold
    best_iteration = 0
    best_score = float('-inf')
    max_round = 500
    # Validation metric needs to improve at least once in every early_stopping_rounds rounds to
    # continue training.
    early_stopping_rounds = 50
    for dtrain, dvalid, _ in train_valid_folds:
        bst.append(xgb.Booster(params, [dtrain, dvalid]))

        # bst.append(xgb.XGBRegressor(params, [dtrain, dvalid]))
        
    # Use CV metric to decide to early stop. CV metric = mean validation accuracy over CV folds
    for iteration_id in range(max_round):
        valid_metric = []
        for fold_id, (dtrain, dvalid, y_valid) in enumerate(train_valid_folds):
            # compute valid_metric_func on the fly
            
            if model_obj.prediction_type == 'time':
                # print('No validation metric identified - assuming Quicker MAE PO')
                # instantiate the validation metric
                # mae_po_calc = MAE_PO()
                # dfvalid = pd.DataFrame(dvalid.get_label(),columns=['time'])
                # dfvalid['event'] = (dfvalid['time']<0).astype(float)
                # dfvalid['time'] = np.abs(dfvalid['time'])

                model_obj.compute_weights_margins(y_valid['event'],y_valid['time'],recompute=True)
                # mae_po_calc.Cum_Ho = predict_baseline_cum_hazard_generic(dftrain)

                def valid_metric_func(y_valid, y_pred):
                    # try:
                    #     y_valid = pd.DataFrame(y_valid)['time'].values
                    # except:
                    #     pass
                    return model_obj.mae_po_reg_metric(model_obj.estimated_risk(y_pred), y_valid)
                
            bst[fold_id].update(dtrain, iteration_id)
            y_pred = bst[fold_id].predict(dvalid)
            valid_metric.append(valid_metric_func(y_valid, y_pred))
        cv_valid_metric = np.mean(valid_metric)
        if cv_valid_metric > best_score:
            best_score = cv_valid_metric
            best_iteration = iteration_id
        elif iteration_id - best_iteration >= early_stopping_rounds:
            # Early stopping
            break

    trial.set_user_attr('num_round', best_iteration)
    trial.set_user_attr('timestamp', time.perf_counter())

    return best_score

def xgb_compute_test_pred(*, model_obj,study, dtrain_valid_combined, dtest,params_df=None,trial_num = 'best'):
    if study:
        if trial_num == 'best':
            best_params = study.best_params
            best_num_round = study.best_trial.user_attrs['num_round']
        else:
            best_params = study.get_trials()[trial_num].params
            best_num_round = 100
    else:
        best_params = params_df
        best_num_round = 100
    
    if best_num_round <100:
        best_num_round = 100
    # print("boost rounds:",best_num_round)

    best_params.update(model_obj.get_base_params())
    model_obj.modelnow = xgb.train(best_params, dtrain_valid_combined,
                                num_boost_round=best_num_round,
                                evals=[(dtrain_valid_combined, 'train_valid'), (dtest, 'test')],
                                verbose_eval=False)
    y_pred = model_obj.modelnow.predict(dtest)
    y_pred_train = model_obj.modelnow.predict(dtrain_valid_combined)

    if trial_num == 'best':
        model_obj.final_model = model_obj.modelnow
        
    return y_pred,y_pred_train 

class WLS_OF:
    def __init__(self, alpha= 1095,beta= 365):
        self.beta = beta
        self.alpha = alpha# this can be modified - or optimised

    # Weighted-Least Squares Objective Function
    def sigmoid_approx(self,y):
        return 2/(1+np.exp((y-self.beta)/self.alpha))

    def gradient(self,predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the gradient squared log error.'''
        y = dtrain.get_label()
        # return -2*(y - predt) / (y**(2*self.S))
        return 2*y**self.S*(y - predt)

    def hessian(self,predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        # return 2 / (y**(2*self.S))
        return 2 * y**self.S

    def squarederror(self,predt: np.ndarray,
                    dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
        
        y = dtrain.get_label()
        self.S = self.sigmoid_approx(y)
        # predt[predt < -1] = -1 + 1e-6
        grad = self.gradient(predt, dtrain)
        hess = self.hessian(predt, dtrain)
        return grad, hess
    
    def mae(self,predt: np.ndarray, dtrain: xgb.DMatrix) -> Tuple[str, float]:
        y = dtrain.get_label()
        self.S = self.sigmoid_approx(y)
        # predt[predt < -1] = -1 + 1e-6
        elements = np.abs(y - predt)
        return 'PyWMAE', float(np.sum(elements) / len(y))

class MAE_PO:

    def compute_weights_margins(self,event,t,recompute=True,save_path = 'data/mae_po_weights.csv'):
        self.event = event
        self.time = t
        if recompute:
            # calculate KM estimator for the whole population
            km_time, km_survival_function, _ = kaplan_meier_estimator(self.event.astype(bool), self.time, conf_type="log-log")

            # import altair as alt
            # df = pd.DataFrame([km_time, km_survival_function],index=['km_time', 'km_survival_function']).T
            # df['type'] = 'All'

            # calculate weights
            w = copy.deepcopy(self.event).astype(float)
            idx = w==0
            interpolator = interp1d(km_time,km_survival_function)
            w[idx,] = 1-interpolator(self.time[idx])
            self.w = w.astype(float)

            # calculate margin - two options: distribution margin / pseudo-observations
            e = np.zeros_like(w)
            for ii in range(len(e)):
                event_aux,time_aux = copy.deepcopy(self.event),copy.deepcopy(self.time)
                event_aux = np.delete(event_aux,ii)
                time_aux = np.delete(time_aux,ii)
                km_time_i, km_survival_function_i, _ = kaplan_meier_estimator(event_aux.astype(bool), time_aux, conf_type="log-log")
                # df2 = pd.DataFrame([km_time_i, km_survival_function_i],index=['km_time', 'km_survival_function']).T
                # df2['type'] = f'PO_{ii}'
                # df = pd.concat([df,df2],axis=0)
                e[ii,] = (km_time*km_survival_function).mean() - (km_time_i*km_survival_function_i).mean()
            self.e = e.reshape(-1,).astype(float)
            # alt.Chart(df.query(f"type in {tuple([k for ki,k in enumerate(df['type'].unique()) if ki<10])}")).mark_line().encode(x='km_time',y='km_survival_function',color='type').save('test.html')
            # # save e and w
            # out = pd.DataFrame([self.e,self.w]).T
            # out.columns=['e','w']
            # out.to_csv(save_path,index=False)
        else:
            try:
                out = pd.read_csv(save_path)
                self.e = out['e'].values.reshape(-1,)
                self.w = out['w'].values.reshape(-1,)
            except:
                self.compute_weights_margins(event,t,recompute=True,save_path = save_path)

    def compute_po_times(self,t):
        return (1-self.event)*self.e+t

    # this objective function should not be used 'as is' because it is not derivable - this is bad for gradient descent
    # def gradient(self,predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    #     '''Compute the gradient for MAE - PO.'''
    #     t = dtrain.get_label()
    #     po_times = self.compute_po_times(self,t)
    #     t_median = calculate_perc_survival_time(self,dtrain.X,predt,f=.5)
    #     return -self.w*(po_times-t_median)/np.abs(po_times-t_median)

    # def hessian(self,predt: np.ndarray, dtrain: xgb.DMatrix) -> np.ndarray:
    #     '''Compute the hessian for MAE - PO.'''
    #     t = dtrain.get_label()
    #     po_times = self.compute_po_times(self,t)
    #     t_median = calculate_perc_survival_time(self,dtrain.X,predt,f=.5)
    #     return 2*self.w/np.abs(po_times-t_median)

    # def mae_po_objective(self,predt: np.ndarray,
    #                 dtrain: xgb.DMatrix) -> Tuple[np.ndarray, np.ndarray]:
    #     return self.gradient(predt, dtrain),self.hessian(predt, dtrain)

    def mae_po_metric(self,predt: np.ndarray, dtrain) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        if isinstance(dtrain,xgb.DMatrix):
            t = dtrain.get_label()
            calltype = 'opt'
        elif isinstance(dtrain,np.ndarray): 
            t = np.array([e[1] for e in dtrain])
            calltype = 'eval'
        else: 
            raise TypeError('The input must be either a xgb.DMatrix or np.ndarray')
        
        # dftrain = pd.DataFrame(dtrain)
        # self.compute_weights_margins(dftrain['event'].values.reshape(-1,),dftrain['time'].values.reshape(-1,),recompute=True)
        self.Cum_Ho = predict_baseline_cum_hazard_generic(dtrain)
        survival_function = predict_survival_function_generic(predt,self.Cum_Ho)
        
        # # this is using the central value
        # n = survival_function.index.shape[0]//2
        # t_median = survival_function.iloc[n,:].values

        # this is using the median value with limit as the last value
        idx_last = survival_function.iloc[-1,:]<.5
        # t_median = np.array([interp1d(survival_function.loc[:,patient],survival_function.index)(.5) for patient in survival_function.loc[:,idx_last].keys()])
        t_median = {patient:interp1d(survival_function.loc[:,patient],survival_function.index)(.5) for patient in survival_function.loc[:,idx_last].keys()}
        t_median.update({patient:survival_function.index[-1] for patient in survival_function.loc[:,~idx_last].keys()})
        t_median = np.array([t for p,t in sorted(t_median.items())])

        elements = self.w*np.abs((1-self.event)*self.e+t-t_median)

        if calltype == 'opt':
            return 'PyMAEPO', float(np.sum(elements) / sum(self.w))
    
        return float(np.sum(elements) / sum(self.w))

    def mae_po_reg_metric(self,predt: np.ndarray, dtrain) -> Tuple[str, float]:
        ''' Root mean squared log error metric.'''
        if isinstance(dtrain,xgb.DMatrix):
            t = dtrain.get_label()
            calltype = 'opt'
        elif isinstance(dtrain,np.ndarray): 
            t = np.array([e[1] for e in dtrain])
            calltype = 'eval'
        else: 
            raise TypeError('The input must be either a xgb.DMatrix or np.ndarray')
        
        t_median = predt

        elements = self.w*np.abs((1-self.event)*self.e+t-t_median)

        if calltype == 'opt':
            return 'PyMAEPO', float(np.sum(elements) / sum(self.w))
    
        return float(np.sum(elements) / sum(self.w))

class XGBoostCox:
    def __init__(self):
        self.method_name = 'XGBoostCox'
        self.direction = 'maximize'
        self.prediction_type = 'survival'
        self.scaler = StandardScaler
    
    def get_params(self, trial):
        eta = trial.suggest_uniform('learning_rate', 0.01, 1.0)
        max_depth = trial.suggest_int('max_depth', 4, 12, step=2)
        min_child_weight = trial.suggest_uniform('min_child_weight', 0.1, 10.0)
        reg_alpha = trial.suggest_uniform('reg_alpha', 0.1, 10)
        reg_lambda = trial.suggest_uniform('reg_lambda', 0.1, 10)
        # n_estimators=trial.suggest_int('n_estimators', 10, 100, step=10)
        return {'eta': eta,
                'max_depth': int(max_depth),
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda}
                # 'n_estimators': n_estimators /////num_parallel_tree
    
    def set_params(self,params_df):
        self.params_df = {'eta': params_df['learning_rate'],
                        'max_depth': int(params_df['max_depth']),
                        'min_child_weight': params_df['min_child_weight'],
                        'reg_alpha': params_df['reg_alpha'],
                        'reg_lambda': params_df['reg_lambda']}
                        # 'n_estimators': int(params_df['n_estimators'])

    def get_base_params(self):
        return {'verbosity': 0,
                'objective': 'survival:cox',
                'tree_method': 'hist',
                'eval_metric': 'cox-nloglik'}
    
    def dmat_builder(self, X, y):
        label = np.array([(e[1] if e[0] else -e[1]) for e in y])
        return xgb.DMatrix(X, label=label)

    def estimated_risk(self, y_pred):
        return np.clip(y_pred,a_min=-10**10,a_max=10**10)

    def train(self, trial, train_valid_folds, valid_metric_func):
        return xgb_train(trial=trial, train_valid_folds=train_valid_folds,
                         model_obj=self, valid_metric_func=valid_metric_func)

    def compute_test_pred(self, study, dtrain_valid_combined, dtest,params_df=None,trial_num = 'best'):
        return xgb_compute_test_pred(model_obj=self,study=study, dtrain_valid_combined=dtrain_valid_combined,
                                     dtest=dtest,params_df=params_df,trial_num=trial_num)
    
    def predict_baseline_hazard(self,df_train):
        # get baseline hazard
        self.Cum_Ho = predict_baseline_cum_hazard_generic(df_train)

    def predict_survival_function(self,X):
        # get the hazard's ratio - (predictions)
        df_pred = self.modelnow.predict(X)
        # get the final survival function 
        survival_function = predict_survival_function_generic(df_pred,self.Cum_Ho)
        return survival_function

class XGBoostMAEPOReg(MAE_PO):
    def __init__(self):
        self.method_name = 'XGBoostRegPO'
        self.direction = 'minimize'
        self.prediction_type = 'time'
        self.survival_calc_mode = 'sigmoid_approx'#'time_translation'
        self.plot_flag = False
        self.scaler = StandardScaler
    
    def get_params(self, trial):
        eta = trial.suggest_uniform('learning_rate', 0.01, 1.0)
        max_depth = trial.suggest_int('max_depth', 4, 12, step=2)
        min_child_weight = trial.suggest_uniform('min_child_weight', 0.1, 10.0)
        reg_alpha = trial.suggest_float('reg_alpha', 0.01, 100, log=True)
        reg_lambda = trial.suggest_float('reg_lambda', 0.01, 100, log=True)
        n_estimators=trial.suggest_int('n_estimators', 10, 100, step=10)
        alpha=trial.suggest_float('alpha', 10, 10000, log=True)
        beta=trial.suggest_float('beta', 10, 10000, log=True)
        wls = WLS_OF(alpha=alpha,beta=beta)
        return {'eta': eta,
                'max_depth': int(max_depth),
                'min_child_weight': min_child_weight,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'obj': wls.squarederror,
                'custom_metric': wls.mae,
                'n_estimators': n_estimators}
    
    def set_params(self,params_df):
        wls = WLS_OF(alpha=params_df['alpha'],beta=params_df['beta'])
        self.params_df = {'eta': params_df['learning_rate'],
                        'max_depth': int(params_df['max_depth']),
                        'min_child_weight': params_df['min_child_weight'],
                        'reg_alpha': params_df['reg_alpha'],
                        'reg_lambda': params_df['reg_lambda'],
                        'n_estimators': int(params_df['n_estimators']),
                        'obj': wls.squarederror,
                        'custom_metric': wls.mae}

    def get_base_params(self):
        # wls = WLS_OF()
        return {'verbosity': 0,
                # 'objective': 'reg:squarederror',
                # 'obj': wls.squarederror,
                'tree_method': 'hist',
                # 'eval_metric': 'mae',
                # 'custom_metric': wls.mae
                # 'eval_metric': self.mae_po_metric
                }
    
    def dmat_builder(self, X, y):
        t = np.array([e[1] for e in y])
        event = np.array([e[0] for e in y])
        self.compute_weights_margins(event,t)
        # label = self.compute_po_times(t)
        return xgb.DMatrix(X, label=t)

    def estimated_risk(self, y_pred):
        return y_pred

    def train(self, trial, train_valid_folds, valid_metric_func):
        return xgb_train(trial=trial, train_valid_folds=train_valid_folds,
                         model_obj=self, valid_metric_func=valid_metric_func)

    def compute_test_pred(self, study, dtrain_valid_combined, dtest,params_df=None,trial_num = 'best'):
        return xgb_compute_test_pred(model_obj=self,study=study, dtrain_valid_combined=dtrain_valid_combined,
                                     dtest=dtest,params_df=params_df,trial_num=trial_num)

    def predict_baseline_hazard(self,df_train):
        # get baseline hazard
        self.Cum_Ho = predict_baseline_cum_hazard_generic(df_train)

    def predict_survival_function(self,X):
        # get the hazard's ratio - (predictions)
        t_median = self.modelnow.predict(X)
        interpolator = interp1d(self.Cum_Ho[0],self.Cum_Ho.index)
        t_median_km = interpolator(.5)

        ############# Option 1: Pseudo-Sigmoid approximation with same median value estimate from baseline survival
        if self.survival_calc_mode == 'sigmoid_approx':
            # now find the sigmoid correction coefficient for the KM estimator
            survkm = np.exp(-self.Cum_Ho).values.reshape(-1,)

            def inv_sig(x, a,Sinf):
                return Sinf+2*(1-Sinf)/(1+np.exp(a * x))

            # Curve fitting & find a
            pars,cov = curve_fit(inv_sig, self.Cum_Ho.index,survkm,p0=[0.001,.3])
            ao,Sinf = pars
            a = ao*(t_median_km/t_median)

            # get all survival functions
            t = np.arange(0,X.get_label().max(),100)
            survival_function = pd.DataFrame(np.array([inv_sig(t,aa,Sinf) for aa in a])).T
            survival_function.index = t
        
        elif self.survival_calc_mode == 'time_translation':
            ############# Option 2: Simple translation in time domain
            C = []
            for tmi in t_median:
                Cum_H = self.Cum_Ho.copy()
                Cum_H.index = np.round(self.Cum_Ho.index+(tmi - t_median_km),0)
                C.append(Cum_H)
            C = pd.concat(C,axis=1).sort_index()

            # get the final survival function 
            survival_function = np.exp(-C)
            survival_function.columns = np.arange(C.shape[1])
        else:
            raise ValueError(f'fsurvival_calc_mode: {self.survival_calc_mode} not understood')

        if self.plot_flag:
            np.random.seed(30)
            dfplot = survival_function.reset_index().melt(id_vars=['index']).rename(columns={'index':'time','variable':'patient','value':'survival_probability'})
            nums = np.random.choice(dfplot['patient'].nunique(),20)
            alt.Chart(dfplot.query(f'patient in {tuple(nums)}')).mark_line().encode(x=alt.X('time').scale(domain=(0, 12000)),
                                                                                    y='survival_probability',
                                                                                    color='patient:N').transform_filter(alt.FieldGTPredicate(field='time', gt=0)).save('data/mytest2.html')

        return survival_function
    
class SkSurvCoxLinear:
    def __init__(self):
        self.method_name = 'SkSurvCoxLinear'
        self.direction = 'maximize'
        self.prediction_type = 'survival'
        self.scaler = StandardScaler

    def get_params(self, trial):
        alpha = trial.suggest_float('alpha', 0.001, 100, log=True)
        return {'alpha': alpha}
    
    def set_params(self,params_df):
        self.params_df = {'alpha': params_df['alpha']}

    def get_base_params(self):
        pass

    def dmat_builder(self, X, y):
        return X, y

    def estimated_risk(self, y_pred):
        return y_pred

    def train(self, trial, train_valid_folds, valid_metric_func):
        params = self.get_params(trial)

        valid_metric = []
        for dtrain, dvalid, _ in train_valid_folds:
            clf = CoxPHSurvivalAnalysis(alpha=params['alpha'], ties='breslow', n_iter=100, tol=1e-9)
            X_train, y_train = dtrain
            X_valid, y_valid = dvalid
            try:
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_valid)
                # valid_metric.append(valid_metric_func(clf,X_valid,y_valid, y_pred))
                valid_metric.append(valid_metric_func(y_valid, y_pred))
            except Exception as e:
                print(e)
                print('metric was non added due to preceeding error')

        cv_valid_metric = np.mean(valid_metric)

        trial.set_user_attr('timestamp', time.perf_counter())

        return cv_valid_metric

    def compute_test_pred(self, study, dtrain_valid_combined, dtest,params_df=None,trial_num='best'):
        if study:
            if trial_num == 'best' or study.best_trial.number==trial_num:
                best_params = study.best_params
            else:
                best_params = study.get_trials()[trial_num].params
        else:
            best_params = params_df
    
        self.modelnow = CoxPHSurvivalAnalysis(alpha=best_params['alpha'], ties='breslow', n_iter=100,
                                            tol=1e-9)
        X_train_valid, y_train_valid = dtrain_valid_combined
        X_test, _ = dtest
        try: 
            self.modelnow.fit(X_train_valid, y_train_valid)
        except Exception as e:
            print(e)
            # subsample the set - COX collinearity fails sometimes
            c = 0
            while c < 10: # try 10 times before failing 
                try: 
                    sample_with_replacement = np.random.choice(X_train_valid.shape[0], size=int(X_train_valid.shape[0]*.9//1), replace=True)
                    self.modelnow.fit(X_train_valid[sample_with_replacement], y_train_valid[sample_with_replacement])
                    c+=1
                except:
                    pass
        y_pred = self.modelnow.predict(X_test)
        y_pred_train = self.modelnow.predict(X_train_valid)
        if trial_num == 'best' or (study and study.best_trial.number==trial_num):
            self.final_model = self.modelnow
        return y_pred,y_pred_train
    
    def predict_baseline_hazard(self,df_train=None):
        self.Cum_Ho = self.modelnow.cum_baseline_hazard_
        # get baseline hazard
        # self.Ho = get_hazard_from_cum_hazard(Cum_Ho)

    def predict_survival_function(self,X):
        survival_function = self.modelnow.predict_survival_function(X[0])
        survival_function = pd.concat([pd.DataFrame(s.y,index=s.x) for s in survival_function],axis=1)
        survival_function.columns = np.arange(1,survival_function.shape[1]+1)
        survival_function.index.name = 'timeline'
        return survival_function
 
def get_train_valid_test_splits(*, X_train_valid, y_train_valid, X_test, y_test, inner_kfold_gen, strat_var, dmat_builder):
    
    dtest = dmat_builder(X_test, y_test)

    # Split remaining data into train and validation sets.
    # Do this 5 times to obtain 5-fold cross validation
    train_valid_ls = []
    dmat_train_valid_combined = dmat_builder(X_train_valid, y_train_valid)
    for train_idx, valid_idx in inner_kfold_gen.split(X_train_valid,strat_var): # with stratification
        dtrain = dmat_builder(X_train_valid[train_idx, :], y_train_valid[train_idx])
        dvalid = dmat_builder(X_train_valid[valid_idx, :], y_train_valid[valid_idx])
        train_valid_ls.append((dtrain, dvalid, y_train_valid[valid_idx]))

    return train_valid_ls, dmat_train_valid_combined, dtest

def run_nested_cv(*, X, y, tau, strat_var, seed, sampler, n_trials, model_obj,dataset=''):

    if model_obj.prediction_type == 'survival':
        def valid_metric_func(y_valid, y_pred):
            try:
                return concordance_index_ipcw(survival_train=y, survival_test=y_valid,
                                            estimate=model_obj.estimated_risk(y_pred), tau=tau)[0]

            except ValueError as e:
                return float('-inf')  # y_pred contains NaN or Inf, ensure that this model gets ignored
    
    elif model_obj.prediction_type == 'time':    
        valid_metric_func = None

    # Nested Cross-Validation with 4-folds CV in the outer loop and 5-folds CV in the inner loop
    start = time.time()
    out = pd.DataFrame()

    # generate from databases - format train/test
    outer_kfold_gen = [(name,np.where(strat_var!=name)[0],np.where(strat_var==name)[0]) for name in np.unique(strat_var)]
    for test_fold_id, train_valid_idx, test_idx in outer_kfold_gen:###### stratified by the label
    # outer_kfold_gen = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)
    # for test_fold_id, (train_valid_idx, test_idx) in enumerate(outer_kfold_gen.split(X, y)):###### stratified by the label
        # train_valid_folds: list of form [(train_set, valid_set), ...], where train_set is used
        #                    for training and valid_set is used for model selection,
        #                    i.e. hyperparameter search
        # dtest: held-out test set; will not be used for training or model selection
        
        inner_kfold_gen = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)#
        # inner_kfold_gen = KFold(n_splits=5, shuffle=True, random_state=seed) # older implementation
        train_valid_folds, dtrain_valid_combined, dtest \
            = get_train_valid_test_splits(X_train_valid=X[train_valid_idx, :],
                                          y_train_valid=y[train_valid_idx],
                                          X_test=X[test_idx, :],
                                          y_test=y[test_idx],
                                          inner_kfold_gen=inner_kfold_gen,
                                          strat_var = strat_var[train_valid_idx],
                                          dmat_builder=model_obj.dmat_builder)
        
        study = optuna.create_study(sampler=sampler, direction=model_obj.direction)
        study.optimize(
            lambda trial: model_obj.train(trial=trial, train_valid_folds=train_valid_folds,
                                           valid_metric_func=valid_metric_func),
            n_trials=n_trials)

        # Use the best hyperparameter set to fit a model with all data points except the
        # held-out test set
        # Then evaluate C-index on the test set
        # _,y_pred_train = model_obj.compute_test_pred(
        #     study=study, dtrain_valid_combined=dtrain_valid_combined, dtest=dtest)
        train_uncensored = np.array([yy[0] for yy in y[train_valid_idx]])
        test_uncensored = np.array([yy[0] for yy in y[test_idx]])

        out = calc_all_metrics(out,study,model_obj,test_fold_id,dtrain_valid_combined, dtest,[train_uncensored,test_uncensored])
        
        if bootstrap_flag:
            out = calc_bootstrap(model_obj,out,X[test_idx, :],y[test_idx],dtrain_valid_combined,train_uncensored,test_uncensored,study,test_fold_id)

        # save the best model of this fold
        with open(f'models/gastro_{model_obj.method_name}_{dataset}_{test_fold_id}.pkl', 'wb') as fid:
            pickle.dump(model_obj, fid)

    end = time.time()
    time_taken = end - start
    print(f'Time elapsed = {time_taken}')
    out['elapsed_time'] = time_taken
    return out

def split_data_add_mice(X,y,strat_var,database_rotation):
    test_idx = (strat_var == database_rotation).reshape(-1,)
    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                 stratify=strat_var, 
    #                                                 test_size=0.3,random_state=seed)
    X_train, X_test, y_train, y_test = X[~test_idx,:],X[test_idx,:],y[~test_idx],y[test_idx]
    
    if mice_imputation:
        # MICE imputation - Multiple Imputation by Chain Equations
        imp_mean = IterativeImputer(random_state=1,n_nearest_features=10,imputation_order='random')
        imp_mean.fit(X_train)
        X_train = imp_mean.transform(X_train)
        X_test = imp_mean.transform(X_test)
    return X_train, X_test, y_train, y_test,test_idx

def select_model_and_features(dataset,method,X0,y00,strat_var0,feature_key,logger,export_feature_names = False):

    print('####################### dataset:', dataset)
    logger.warning(f'computing: {dataset}')
    if dataset == 'allfeatures':
        feature_names = feature_key.loc[feature_key['type'].isin(['raw']),'feature'].to_list()
    elif dataset == 'classes+decay':
        feature_names = feature_key.loc[feature_key['type'].isin(['class','decay','longitudinal']),'feature'].to_list()
    elif dataset == 'classes':
        feature_names = feature_key.loc[feature_key['type'].isin(['class','longitudinal']),'feature'].to_list()
    elif dataset == 'classes_eq':
        feature_names = feature_key.loc[feature_key['type'].isin(['class_eq','longitudinal']),'feature'].to_list()
    elif dataset == 'classes_neq':
        feature_names = feature_key.loc[feature_key['type'].isin(['class_neq','longitudinal']),'feature'].to_list()
    elif dataset == 'classes_slope':
        feature_names = feature_key.loc[feature_key['type'].isin(['class_slope','longitudinal']),'feature'].to_list()
    elif dataset == 'classes_neqm':
        feature_names = feature_key.loc[feature_key['type'].isin(['class_neqm','longitudinal']),'feature'].to_list()
    elif dataset == 'demo+classes_neqm':
        feature_names = feature_key.loc[feature_key['type'].isin(['demo','class_neqm','longitudinal']),'feature'].to_list()
    elif dataset == 'demo':
        feature_names = feature_key.loc[feature_key['type'].isin(['demo']),'feature'].to_list()
    elif dataset == 'decayfeatures':
        feature_names = feature_key.loc[feature_key['type'].isin(['decay','demo']),'feature'].to_list()
    elif dataset == 'rawfeatures':
        feature_names = feature_key.loc[feature_key['type'].isin(['raw','demo']),'feature'].to_list()
    elif dataset == 'indiv':
        feature_names = feature_key.loc[feature_key['type'].isin(['class_neqm','longitudinal','demo']),'feature'].to_list()
    elif dataset == 'encals':
        # currently this block of code will not work
        X00 = X0.copy()
        X00.loc[:,'C9orf72'] = X00.loc[:,'C9orf72'].fillna(0)
        g = X00.isna().sum()
        feature_names = feature_key.loc[feature_key['type'].isin(['encals']),'feature'].to_list()
        cols_for_imputing = [k for k in g[(g==0).values].index if k not in feature_names]+feature_names
        imputer_df = X00.loc[:,cols_for_imputing]

        # MICE imputation - Multiple Imputation by Chain Equations
        imp_mean = IterativeImputer(random_state=1,n_nearest_features=10,imputation_order='random')
        aux = imp_mean.fit_transform(imputer_df)

        X00 = aux[:,-7:]
    
    else:
        raise Exception(f'Unknown dataset {dataset}')
    
    # assign features
    X00 = X0.loc[:,feature_names].astype(float)

    # if mice_imputation:

    #     # MICE imputation - Multiple Imputation by Chain Equations
    #     imp_mean = IterativeImputer(random_state=1,n_nearest_features=10,imputation_order='random')
    #     aux = imp_mean.fit_transform(X00)

    #     X00 = pd.DataFrame(aux,columns=X00.columns)

    # feature_names = list(X00.keys()) if dataset!='encals' else encals_cols
    
    X00 = X00.values if dataset!='encals' else X00
    y = y00.copy()
    strat_var = strat_var0.copy()

    print('####################### method:', method)
    logger.info(f'computing method: {method}')
    if method == 'XGBoostCox':
        model_obj = XGBoostCox()
        X00 = model_obj.scaler().fit_transform(X00)
        X = X00.copy()
    elif method == 'XGBoostMAEPOReg':
        model_obj = XGBoostMAEPOReg()
        X00 = model_obj.scaler().fit_transform(X00)
        X = X00.copy()
    elif method == 'SkSurvCoxLinear':
        model_obj = SkSurvCoxLinear()
        # scale features
        X00 = model_obj.scaler().fit_transform(X00)
        X = X00.copy()        
        X,y,strat_var,cols_to_keep = remove_nans(X00,y00,strat_var0,col_threshold = 4000)
        feature_names = list(np.array(feature_names)[cols_to_keep])
    else:
        raise Exception(f'Unknown method {method}')
   
    if export_feature_names:
        return model_obj,X,y,strat_var,feature_names
    
    return model_obj,X,y,strat_var,feature_names

def hyperopt_all_methods(dataset,method,X0, y00, tau, strat_var0, feature_key,logger, seed=1,n_trials=2):
    
    # try:
    if True:
        # select model
        model_obj,X,y,strat_var,_ = select_model_and_features(dataset,method,X0,y00,strat_var0,feature_key,logger)

        out = run_nested_cv(X=X, y=y, tau=tau, strat_var=strat_var,seed=seed,
                    sampler=RandomSampler(seed=seed),
                    n_trials=n_trials,
                    model_obj=model_obj,dataset=dataset)

        out['dataset'] = dataset
        out['method'] = method
        out['n_features'] = dict_number_of_features[dataset]
        out['n_patients'] = X.shape[0]
        out['processed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # O.append(out)
        out.to_csv(f'logs/{dataset}_{method}_{uuid4()}.csv')
        logger.info(f'{dataset}_{method}_{uuid4()} finished successfully')
        return out
    # except Exception as e:
    #     logger.error(f'{dataset}_{method}_{uuid4()} failed')
    #     logger.error(e)
    #     print(e)
    #     return pd.DataFrame([])

def run_indiv_pred(xi,x,row,model_obj,y,strat_var,feature_names):
    print('Computing individual predictions:',feature_names[xi],f"{xi+1}/{len(feature_names)}")
    x = x.reshape(-1,1)
    dfaux,dfsummary,dfpred = train_all_methods(row,model_obj,x,y,strat_var)
    dfaux['feature'] = feature_names[xi]
    dfsummary['feature'] = feature_names[xi]
    dfpred['feature'] = feature_names[xi]
    return dfaux,dfsummary,dfpred

def train_all_methods_wrapper(row,X0,y00,tau,strat_var0,feature_key,logger):
    
    model_obj,X,y,strat_var,feature_names = select_model_and_features(row.dataset,row.method,X0,y00,strat_var0,feature_key,logger)
    
    if row.dataset!='indiv':    
        dfaux,dfsummary,dfpred = train_all_methods(row,model_obj,X,y,strat_var)
    else:
        from multiprocessing import Pool
        
        with Pool(14) as p:
            input_data = ((xi,x,row,model_obj,y,strat_var,feature_names) for xi,x in enumerate(X.T))
            res = p.starmap(run_indiv_pred, input_data)

        # decode the results
        dfaux,dfsummary,dfpred = [],[],[]
        for r in res:
            dfsummary.append(r[1])
            dfaux.append(r[0])
            dfpred.append(r[2])
        dfaux = pd.concat(dfaux,axis=0,ignore_index=True)
        dfsummary = pd.concat(dfsummary,axis=0,ignore_index=True)
        dfpred = pd.concat(dfpred,axis=0,ignore_index=True)
    # dfaux.to_csv('daux.csv'),dfsummary.to_csv('dfsummary.csv'),dfpred.to_csv('dfpred.csv')
    return dfaux,dfsummary,dfpred

def train_all_methods(row,model_obj,X,y,strat_var):

    Daux,Dsummary,Dpred = [],[],[]
    
    database_rotation = row['test_fold']
    print(f'########## rotation: {database_rotation}')
    start = time.time() 

    # perform split
    X_train, X_test, y_train, y_test, test_idx = split_data_add_mice(X,y,strat_var,database_rotation)

    # set parameters
    model_obj.set_params(row)

    additive_name = '_mice' if mice_imputation else ''
    
    if test_only:
        # reload model
        with open(f'models/gastro_{row.method}_{row.dataset}_{database_rotation}{additive_name}.pkl', 'rb') as fid:
            model_obj = pickle.load(fid)
        
        # build matrices
        dtest = model_obj.dmat_builder(X_test, y_test)
        dtrain_valid_combined = model_obj.dmat_builder(X_train, y_train)
        if row.method == 'SkSurvCoxLinear':
            dtest = dtest[0]
            dtrain_valid_combined = dtrain_valid_combined[0]

        # get predictions
        y_pred = model_obj.modelnow.predict(dtest)
        y_pred_train = model_obj.modelnow.predict(dtrain_valid_combined)
    
    else:
        
        # build matrices
        dtest = model_obj.dmat_builder(X_test, y_test)
        dtrain_valid_combined = model_obj.dmat_builder(X_train, y_train)

        # compute predictions
        y_pred,y_pred_train = model_obj.compute_test_pred(None, dtrain_valid_combined, dtest,params_df=model_obj.params_df)

        # save model            
        with open(f'models/gastro_{row.method}_{row.dataset}_{database_rotation}{additive_name}.pkl', 'wb') as fid:
            pickle.dump(model_obj, fid)

    # kernel = row.kernel if isinstance(row.kernel,str) else ''
    # pickle.dump(model_obj, open('data/'+row.method+row.dataset+kernel+'.sav', 'wb'))
    # pickle.dump(model_obj, open('data/'+row.method+row.dataset+'.sav', 'wb'))
    train_idx = np.where(~test_idx)[0]
    test_idx = np.where(test_idx)[0]
    train_uncensored = np.array([yy[0] for yy in y[train_idx]])
    test_uncensored = np.array([yy[0] for yy in y[test_idx]])

    # compute metrics
    # cindex,concordant,discordant,tied_risk,tied_time = concordance_index_ipcw(survival_train=y_train, survival_test=y_test,
    #                                     estimate=model_obj.estimated_risk(y_pred), tau=tau)
    
    dfsummary = calc_all_metrics(pd.DataFrame(),None,model_obj,None,dtrain_valid_combined, dtest,[train_uncensored,test_uncensored])
    dfsummary['test_rotation'] = database_rotation
    dfsummary['method'] = row.method
    dfsummary['dataset'] = row.dataset
    dfsummary['n_features'] = dict_number_of_features[row.dataset]
    dfsummary['n_patients'] = X.shape[0]
    dfsummary['processed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    end = time.time()
    time_taken = end - start
    print(f'Time elapsed = {time_taken}')
    dfsummary['elapsed_time'] = time_taken

    # bootstrapping
    if bootstrap_flag:
        dfsummary = calc_bootstrap(model_obj,dfsummary,X_test,y_test,dtrain_valid_combined,train_uncensored,test_uncensored)
        # try: 
        if True:

            if row.method != 'XGBoostMAEPOReg':
                model_obj.predict_baseline_hazard(pd.DataFrame(y_train))
                # calculate the rest
                y_pred_test = calculate_perc_survival_time(model_obj,X_test, y_test,f=f_threshold)
                y_pred_train = calculate_perc_survival_time(model_obj,X_train, y_train,f=f_threshold)
            else: 
                y_pred_test = y_pred.copy()
            
            y_test_out = np.array([n[1] for n in y_test])
            offsets = np.arange(0,1080,5)
            accuracies = [cal_predintol(y_test_out,y_pred_test,offset=offset) for offset in offsets]
            dfaux = pd.DataFrame([offsets,accuracies]).T.rename(columns={0:'offset',1:'PredInTol'})
            
            # dfaux['Cindex'] = cindex
            dfaux['method'] = row.method
            dfaux['dataset'] = row.dataset
            dfaux['test_rotation'] = database_rotation

            # store all predictions
            dfpred = get_df_pred(y_train,y_test,y_pred_train,y_pred_test,train_idx,test_idx,database_rotation,row.method,row.dataset)

            # dfaux['kernel'] = row.kernel
            # df = pd.concat([df,dfaux],axis=0,ignore_index=True)

            # perform a threshold optimization
            # Acc = []
            # thresholds = np.arange(0.1,1,.1)
            # for f in thresholds:
            #     print(f)
            #     y_pred_test = calculate_perc_survival_time(model_obj,X_test, y_test,f=f)
            #     accuracies = [cal_accuracy(y_test_out,y_pred_test,offset=offset) for offset in offsets]
            #     Acc.append(accuracies)
            # Acc = pd.DataFrame(np.array(Acc).T,columns=thresholds).melt()

            # alt.Chart(Acc)


            # dfaux.to_csv(f'logs/simul_{time.strftime("%Y%m%d-%H%M%S")}.csv',index=False)
        # except Exception as e:
        #     print(e)
        #     dfaux = pd.DataFrame()
        Daux.append(dfaux.copy())
        Dsummary.append(dfsummary.copy())
        Dpred.append(dfpred.copy())
        # dfsummary.to_csv(f'summary_{database_rotation}_{row.method}_{row.dataset}.csv',index=False)
        # dfpred.to_csv(f'dfpred_{database_rotation}_{row.method}_{row.dataset}.csv',index=False)
        # dfaux.to_csv(f'dfaux_{database_rotation}_{row.method}_{row.dataset}.csv',index=False)
    dfaux = pd.concat(Daux,axis=0,ignore_index=True)
    dfsummary = pd.concat(Dsummary,axis=0,ignore_index=True)
    dfpred = pd.concat(Dpred,axis=0,ignore_index=True)
    return dfaux,dfsummary,dfpred
    # print(f'Done! row:{ri+1}/{models_df.shape[0]}')




# simulation.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# This file simulates different scenarios
import pickle
from utils.utils_censor import select_model_and_features,load_data,calc_all_metrics
from utils.utils import load_death_features,cal_metrics
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import time
import altair as alt
from utils.constants import *
from utils.utils_censor import *

logging.basicConfig(filename=f'logs/simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}_pred.log',level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Create a logger
logger = logging.getLogger(__name__)

simulations = [5]
recompute_flag = False

def gastro_ever(df_combo):
    df_combo['pred_gastro_ever'] = df_combo['death_pred']>df_combo['gastrostomy_pred']
    df_combo['gt_gastro_ever'] = df_combo['gastro_bool']
    df_combo['acc_gastro_ever'] = df_combo['gt_gastro_ever'] == df_combo['pred_gastro_ever']
    return df_combo['acc_gastro_ever'].sum()/df_combo.shape[0]


if 0 in simulations:
# Simulation 0: weight calculation from weight_delay_dataset2 - imperial-als/fit_relu_weights_sandbox.py
    bootstraps = 100
    df = pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/weight_delay_dataset2.csv')
    df2 =  pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv')
    df2 = df2.loc[:,['Database','id','tcens','cens']].rename(columns={'id':'numid'})
    df = df.query('threshold == 0.1')

    df = df2.merge(df,on='numid')

    df_unc = df.query('cens == 0')

    # calculate metrics of the two models
    def calc_metrics_wrapper(L,li):
        L = L.reset_index(drop=True)
        n = L.shape[0]
        D = []
        for _ in range(bootstraps):
            idx = np.arange(n)
            np.random.shuffle(idx)
            idx = idx[:int(n*.5)]
            metrics_test = cal_metrics(L.loc[idx,'tcens'],L.loc[idx,'delay'])
            dfnow = pd.DataFrame(metrics_test,index=[0])
            D.append(dfnow)
        dfnow = pd.concat(D,axis=0,ignore_index=True)
        dfnow = dfnow.describe().reset_index()
        dfnow = pd.DataFrame([dfnow.query('index=="mean"').iloc[:,1:].astype(float).values[0],
                        (dfnow.query('index=="mean"').iloc[:,1:].astype(float).values - dfnow.query('index=="std"').iloc[:,1:].astype(float).values)[0],
                      (dfnow.query('index=="mean"').iloc[:,1:].astype(float).values + dfnow.query('index=="std"').iloc[:,1:].astype(float).values)[0]],
                      index=['mean','lb','ub'],
                      columns=dfnow.iloc[:,1:].columns)
        dfnow = dfnow.T
        dfnow['test_rotation'] = li
        return dfnow
    dfout = pd.concat([calc_metrics_wrapper(L,li) for li,L in df_unc.groupby('Database')],axis=0)
    dfout.to_csv('data/results_weight_naive_model.csv')

if 5 in simulations:
    # descriptive analysis
    df2 =  pd.read_csv('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv')
    idx = ~df2.loc[:,['Death_Date', 'Outcome_Date']].isna().all(axis=1)

    df2.loc[idx, 'last_follow_up'] = np.nan

    idx = df2.loc[:,['Death_Date', 'Outcome_Date']].notna().all(axis=1)

    df2.loc[idx, 'Death_Date'] = np.nan
    idx = ~(df2.loc[:,['Death_Date', 'last_follow_up',  'Outcome_Date']]<0).any(axis=1)
    df2 = df2.loc[idx,:]

    df2.loc[:,['Death_Date', 'last_follow_up',  'Outcome_Date']].describe()

    static_num_keys = ['age_at_onset', 'diagnostic_delay_months']
    df2['diagnostic_delay_months'] = pd.cut(df2['diagnostic_delay_months'],bins=[0,5,10,15,np.inf],labels=['<5','5-10','10-15','15-20'])
    df2['age_at_onset'] = pd.cut(df2['age_at_onset'],bins=[0,20,50,60,70,np.inf],labels=['<20','20-50','50-60','60-70','>70'])
    static_cat_keys = ['sex','site_onset', 'el_escorial', 'Phenotype']
    keys = ['Database','Death_Date', 'last_follow_up',  'Outcome_Date']
    df2['label'] = np.where(df2.loc[:,keys[1:]].notna())[1]
    df2['label'] = df2['label'].map({0:'death',1:'loss of follow up',2:'gastrostomy'})
    groupvar = ['Database','label']
    # df2.groupby(groupvar)[static_num_keys].mean()

    df2['Phenotype'] = df2['Phenotype'].apply(lambda x:str(x).split('_')[0])

    el_escorial_map = {'Probable Laboratory Supported':'Probable',
    'Definite':'Definite', 
    'Probable':'Probable',
        'Possible':'Possible', 
        'Clinically Definite':'Definite', 
        'Clinically Probable':'Probable',
        'Clinically lab supported':'Definite', 
        'Clinically Suspected':'Possible',
        'Clinically Possible':'Possible', 
        'ALS UMN predominant and monomelinic':'Definite'}

    df2['el_escorial'] = df2['el_escorial'].map(el_escorial_map)
    site_onset_map = {'Spinal':'Spinal', 'Bulbar':'Bulbar', 'Generalised':"Other", 'Respiratory':"Other", 'Other':"Other"}
    df2['site_onset'] = df2['site_onset'].map(site_onset_map)
    A = []
    for cat in static_cat_keys+static_num_keys:
        aux = df2.groupby(groupvar)[cat].value_counts().reset_index()
        aux['category'] = cat
        aux = aux.rename(columns={cat:'variable'})
        aux = aux.merge(aux.groupby(groupvar)['count'].sum().reset_index(),
                on=groupvar,
                suffixes=['','total'])
        aux['pct'] = aux['count']/aux['counttotal']
        A.append(aux)

    A = pd.concat(A,axis=0)

    for ci,cat in enumerate(A['category'].unique()):
        chartnow = alt.Chart(A).mark_bar().encode(x = alt.X('pct',stack=True),
                                    y=alt.Y('Database').title(None),
                                    color=alt.Color('variable').scale(scheme='category20').legend(orient='bottom',title=cat),
                                    column=alt.Column('label').title(None)).transform_filter(alt.FieldEqualPredicate(field='category',
                                                                                            equal=cat))
        if ci==0:
            chart = chartnow
        else:
            chart &= chartnow
                                                                                            
    chart.resolve_scale(color='independent').save('figures/description.html')
    # compute statistics 
    # for each database and variable - measure if they were different accorss labels
    

# column label, row category by category, x count, y  Database
if 1 in simulations:
    # Simulation 1: comparison between death vs gastrostomy model for new metric definition
    # paths = ['models/death_XGBoostReg_ArQ.pkl','models/gastro_XGBoostMAEPOReg_classes+decay_PROACT.pkl']
    paths = ['models/gastro_XGBoostMAEPOReg_classes_ArQ.pkl']
    # with open(paths[0], 'rb') as fid:
    #     clf_death = pickle.load(fid)

    with open(paths[0], 'rb') as fid:
        clf_gastro = pickle.load(fid)
        
    # load data - gastro
    fixed_args = load_data()
    _,Xg,yg,strat_var = select_model_and_features('classes','XGBoostMAEPOReg',fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)

    # load data - death 
    Xd, yd,dffeat,df = load_death_features('/Users/juandelgado/Desktop/Juan/code/imperial/imperial-als/data/master_final_0807.csv',
                                        'data/encals_overall_survival_pred_final.csv')
    
    yd = yd[dffeat['tcens']>0]
    # compute gastro predictions 
    dmat = clf_gastro.dmat_builder(Xg,yg)
    y_pred_gastro = clf_gastro.final_model.predict(dmat)

    # load death predictions
    # dmat = clf_death.dmat_builder(Xd[dffeat['tcens']>0],yd)
    # y_pred_death = clf_death.final_model.predict(dmat)
    y_pred_death = df.loc[df['tcens']>0,'OUT'].values

    ### for own death model and best gastro
    yd0 = np.array([d[0] for d in yd])
    yd1 = np.array([d[1] for d in yd])
    yg0 = np.array([d[0] for d in yg])
    yg1 = np.array([d[1] for d in yg])
    df_combo = pd.DataFrame([y_pred_gastro,y_pred_death,yg0,yg1,yd0,yd1],index=['gastrostomy_pred','death_pred','gastro_bool','gastrostomy_gt','death_bool','death_gt']).T    
    acc_xg = gastro_ever(df_combo)
    print("gastro ever accuracy (XGboost):",np.round(acc_xg,3)*100,'%')

    ## for encals death model and best gastro
    df = df.loc[df['tcens']>0,['OUT','Death_Date']]
    df_combo = pd.DataFrame([y_pred_gastro,df['OUT'].values*365.25/12,yg0,yg1,yd0,yd1],index=['gastrostomy_pred','death_pred','gastro_bool','gastrostomy_gt','death_bool','death_gt']).T    
    # df_combo = df_combo.loc[~df_combo.loc[:,['death_bool','gastro_bool']].all(axis=1),:]
    acc_encals = gastro_ever(df_combo)
    print("gastro ever accuracy (ENCALS):",np.round(acc_encals,3)*100,'%')

    df_combo['lbl'] = df_combo['death_bool'].apply(lambda x:'Death' if x else 'Loss follow up')
    df_combo.loc[df_combo.loc[:,'gastro_bool'].values==True,'lbl'] = 'Gastrostomy'
    df_combo['gastrostomy_pred'] = df_combo['gastrostomy_pred'].astype(float).clip(lower=0, upper=13000)
    df_combo['death_pred'] = df_combo['death_pred'].astype(float).clip(lower=0, upper=13000)
    
    df_combo['height'] = 1300
    base = alt.Chart(df_combo)
    chart1 = base.mark_circle().encode(x=alt.X('gastrostomy_pred',axis=alt.Axis(style='log')),
                                                y=alt.Y('gastrostomy_gt:Q',axis=alt.Axis(style='log')),
                                                color='lbl:N')
    chart2 = base.mark_circle().encode(x=alt.X('death_pred',axis=alt.Axis(style='log')),
                                                y=alt.Y('death_gt:Q',axis=alt.Axis(style='log')),
                                                color='lbl:N'
                                                )
    line_ref = base.mark_rule(color='black').encode(
        x=alt.value(0),
        x2=alt.value('width'),
        y=alt.value('height'),
        y2=alt.value(0)
    )
    ((chart1+line_ref).facet(row='lbl:N') | (chart2+line_ref).facet(row='lbl:N')).save('figures/death_gastro_scatterb.html')

    # likelihood inference
    c = list((df_combo['gastrostomy_pred']-df_combo['death_pred']).quantile([.33,.66]))
    df_combo['gastrostomy_proba'] = pd.cut(df_combo['gastrostomy_pred']-df_combo['death_pred'],[-np.inf]+c+[np.inf], 
                                            labels=["Likely", "Possibly", "Unlikely"])
    print(f'the tertile points are:{c}')

    g = df_combo.groupby('gastrostomy_proba')
    print('probabilities of gastro by proba dist:',g['lbl'].value_counts()/g['lbl'].count())
    print('probabilities of gastro by label dist:',g['lbl'].value_counts()/df_combo['lbl'].value_counts())
    aux = (g['lbl'].value_counts()/df_combo['lbl'].value_counts()).reset_index()

    aux.to_csv('data/gastro_ever_dist2.csv',index=False)

    # proba_palette = {'Likely':'#5AA3D1','Unlikely':'#4076B3','Possibly':'#9BC9E2'}
    proba_palette = {'Possibly':'#5AA3D1','Unlikely':'#4076B3','Likely':'#9BC9E2'}

    # scatter plot - just one
    base = alt.Chart(df_combo)
    chart3 = base.mark_circle().encode(x=alt.X('gastrostomy_pred',axis=alt.Axis(style='log')).title('predicted days to gastrostomy'),
                                                y=alt.Y('death_pred:Q',axis=alt.Axis(style='log')).title('predicted days to death'),
                                                color=alt.Color('gastrostomy_proba:N').title('Gastrostomy probability').scale(domain=list(proba_palette.keys()),
                                                                                                                            range=list(proba_palette.values())))
    # 
    chart3.configure_legend(orient='bottom').save('figures/death_gastro_scatter2b.html')

        # bar plot
    aux['count'] = np.round(aux['count'],2)
    base = alt.Chart(aux).encode(y=alt.Y('count').scale(domain=[0, 1]).title('proportion'),
                                        x=alt.X('lbl').title(None).sort(['Gastrostomy', 'Death','Loss follow up' ]))
    bars = base.mark_bar().encode(color=alt.Color('gastrostomy_proba').title('Gastrostomy probability').scale(domain=list(proba_palette.keys()),
                                                                                                                            range=list(proba_palette.values())),
                                tooltip = ['count','lbl','gastrostomy_proba'])
    # text = base.mark_text().encode(text='count',y=alt.Y('sum(count)'))
    (bars).configure_legend(disable=True).properties(width=150).save('figures/proportionsb.html')


    # now plot the probability curves side by side
    dmat = clf_gastro.dmat_builder(Xg,yg)
    
    clf_gastro.predict_baseline_hazard(pd.DataFrame(yg))
    surv_curve = clf_gastro.predict_survival_function(dmat)

    # dmat = clf_death.dmat_builder(Xd[dffeat['tcens']>0],yd)
    # y_pred_death = clf_death.final_model.predict(dmat)

    # dfcombo is for encals already
    cols = ['gastrostomy_pred', 'death_pred', 
    'gastrostomy_gt','death_gt']
    # base = alt.Chart(df_combo.reset_index()).mark_circle()
    # line1 = base.encode(x='index',y='death_pred',color=alt.value('red'))
    # line2 = base.encode(x='index',y='gastrostomy_pred')
    # (line1+line2).save('aa.html')

from lifelines import KaplanMeierFitter

def get_survival_fcn(kmf,name):
    df = kmf.survival_function_
    df['label'] = name
    df = df.reset_index()
    df.columns = ['time','proportion','label']
    return df

def calculate_KM(T,E,name):
    kmf_ = KaplanMeierFitter()
    kmf_.fit(T, E, label=name)
    return get_survival_fcn(kmf_,name)

kmf = [calculate_KM(df_combo['gastrostomy_gt'],df_combo['gastro_bool'],'Gastrostomy True')]
kmf.append(calculate_KM(df_combo['death_gt'],df_combo['death_bool'],'Death True'))
kmf.append(calculate_KM(df_combo['gastrostomy_pred'],df_combo['gastro_bool'],'Gastrostomy Pred'))
kmf.append(calculate_KM(df_combo['death_pred'].fillna(0.),df_combo['death_bool'],'Death Pred'))

kmf = pd.concat(kmf,axis=0)

kmf['proportion'] = 1 - kmf['proportion']

palette_gastr = {'Gastrostomy True':'#9BC9E2','Gastrostomy Pred':'#4076B3','Death True':'#db7d83','Death Pred':'#c26ac8'}


alt.Chart(kmf).mark_line().encode(x='time',y='proportion',
                                  color=alt.Color('label').scale(domain=list(palette_gastr.keys()),
                                                                 range=list(palette_gastr.values()))).configure_legend(orient='bottom').save('figures/km_plot_predb.html')

# from lifelines.utils import median_survival_times
# from lifelines.datasets import load_waltons
# import matplotlib.pyplot as plt
# from lifelines.plotting import add_at_risk_counts

# plt.figure()
# ax = plt.subplot(111)

    # add_at_risk_counts(*kmf,  
    #                 ax=ax,
    #                 rows_to_show=["At risk"])

    # plt.tight_layout()
    # plt.show()


# Simulation 2: Sensitivity analysis - forward propagation sensitivity
# load the data
if 2 in simulations:
    if recompute_flag:
        O = []
        for model in ['XGBoostMAEPOReg','XGBoostCox','SkSurvCoxLinear']:
            for dataset in ['classes','encals','decayfeatures','classes+decay']:
                tic = time.time()
                fixed_args = load_data()
                _,Xg,yg,strat_var = select_model_and_features(dataset,model,fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
                fixed_args = load_data(suffix='2')
                _,Xg2,yg2,_ = select_model_and_features(dataset,model,fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
                fixed_args = load_data(suffix='4')
                _,Xg4,yg4,_ = select_model_and_features(dataset,model,fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
                fixed_args = load_data(suffix='6')
                _,Xg6,yg6,_ = select_model_and_features(dataset,model,fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger)
                print('load the data:',model,dataset,time.time() - tic)
                
                for database in ['PROACT','ArQ','IDPP']:
                    tic = time.time()
                    # load clf gastro
                    with open(f'models/gastro_{model}_{dataset}_{database}.pkl', 'rb') as fid:
                        clf_gastro = pickle.load(fid)

                    # evaluate on MedianAE of prediction based on increasing timepoints
                    for X,y,n_points in [[Xg,yg,'all'],[Xg2,yg2,'2'],[Xg4,yg4,'4'],[Xg6,yg6,'6']]:
                        dmat = clf_gastro.dmat_builder(X,y)
                        y_pred_gastro = clf_gastro.final_model.predict(X) if model == 'SkSurvCoxLinear' else clf_gastro.final_model.predict(dmat)
                        test_idx = np.where(strat_var==database)[0]
                        train_idx = np.where(strat_var!=database)[0]
                        outnow = calc_all_metrics(pd.DataFrame(),None,clf_gastro,0,1300,X,y,y_pred_gastro[test_idx],y_pred_gastro[train_idx],train_idx,test_idx)
                        outnow['model'] = model
                        outnow['features'] = dataset
                        outnow['n_points'] = n_points
                        outnow['test_rotation'] = database
                        O.append(outnow)
                    print('compute forward performance data:',model,dataset,database,time.time() - tic)

        out = pd.concat(O,axis=0,ignore_index=True)
        out.to_csv('data/forward_propagation_results2.csv',index=False)
    else:
        out = pd.read_csv('data/forward_propagation_results2.csv')

    selected_cols = {'PredIn90_test_uncensored':'PredIn90','MedianAE_test_uncensored':'MedianAE','Cindex_test_uncensored':'Cindex'}
    outsimple = out.groupby(['model','features','n_points'])[list(selected_cols.keys())].mean().reset_index(drop=False)
    outsimple = outsimple.melt(id_vars=['model','features','n_points'])
    outsimple['variable'] = outsimple['variable'].map(selected_cols)
        
    base = alt.Chart(outsimple).encode(x='n_points',y=alt.Y('value').title(None),color=alt.Color('model').scale(domain=list(palette.keys()),range=list(palette.values())),
                                    tooltip=['n_points','value','model','features','variable'])
    dots = base.mark_circle()
    lines = base.mark_line().encode(strokeDash='features')
    (dots+lines).transform_filter(alt.FieldOneOfPredicate('features',oneOf=['encals','classes+decay'])).facet(column=alt.Column('variable',title=None)).resolve_scale(y='independent').save('figures/forward_prop_sensitivity.html')

    # Simulation 3: Sensitivity analysis - feature sensitivity - Permutation importance
if 3 in simulations:
    # permutation importance    
    with open('models/gastro_XGBoostMAEPOReg_classes+decay_PROACT.pkl', 'rb') as fid:
        clf_gastro = pickle.load(fid)
        
    fixed_args = load_data()
    _,Xg,yg,strat_var,feature_names = select_model_and_features('classes+decay','XGBoostMAEPOReg',fixed_args[0],fixed_args[1],fixed_args[3],fixed_args[4],logger,export_feature_names=True)
    if recompute_flag:
        ygt = np.array([yy[1] for yy in yg]).astype(float)
        dmat = clf_gastro.dmat_builder(Xg,yg)
        y_pred_gastro = clf_gastro.final_model.predict(dmat)

        out = pd.DataFrame(cal_metrics(ygt,y_pred_gastro),index=[0])

        n_repeats = 3
        for fi in range(Xg.shape[1]):
            for ri in range(n_repeats):
                Xgnow = Xg.copy()
                np.random.shuffle(Xgnow[:,fi])
                dmat = clf_gastro.dmat_builder(Xgnow,yg)
                y_pred_gastro = clf_gastro.final_model.predict(dmat)
                outnow = pd.DataFrame(cal_metrics(ygt,y_pred_gastro),index=[0])
                outnow['repeat'] = ri
                outnow['feature'] = fi
                out = pd.concat([out,
                        outnow],axis=0,ignore_index=True)
                print(ri,fi)
        out.to_csv('data/permutation_importance.csv',index=False)
    else:
        out = pd.read_csv('data/permutation_importance.csv')

    out['feature'] = out['feature'].fillna('all').astype(str)

    out2 = out.groupby('feature').mean().drop(columns=['repeat']).reset_index()
    ref = out2.query('feature == "all"')['MedianAE'].values[0]
    out2['MedianAE_Err'] = np.abs(out2['MedianAE']-ref)/ref
    out2 = out2.sort_values('MedianAE_Err',ascending=False)
    idxs = out2.query("MedianAE_Err>0.05")['feature'].values.astype(float).astype(int)
    print('Top importance features > 5% MedianAE change:',np.array(feature_names)[idxs])

    out3 = out2.query('feature != "all"')
    out3.feature = np.array(feature_names)[out3.feature.astype(float).astype(int)]
    out3 = out3.sort_values('MedianAE_Err',ascending=False)
    out3['significant'] = out3['MedianAE_Err']>0.05
    out3['MedianAE_Err%'] = out3['MedianAE_Err']/out3['MedianAE_Err'].sum()

    base = alt.Chart(out3).encode(y=alt.Y('feature:N').sort('-x'),x='MedianAE_Err%:Q',tooltip=['MedianAE_Err%','MedianAE_Err','feature'])
    (base.mark_bar()+base.mark_text(align='left', dx=2).encode(text=alt.Text('MedianAE_Err%:Q', format='.2f'))).transform_filter(alt.FieldGTPredicate(field='MedianAE_Err',
                                                        gt=0.00)).save('figures/PI.html')
    # ,
    # top features - 'classes+decay','XGBoostMAEPOReg'
    # ['b_days', 'a_% predicted', 'b_% predicted', 'age_at_onset',
    #     'b_ALSFRS_Total', 'bulbar_subscore_4', 'diagnostic_delay_months',
    #     'ALSFRS_bulbar_Slope_Onset_to_FirstALSFRS',
    #     'ALSFRS_Slope_Onset_to_FirstALSFRS']


    # Simulation 4: Sensitivity analysis - missingness sensitivity
if 4 in simulations:

# load data, models
    if recompute_flag:
        Out = pd.DataFrame()
        fixed_args = load_data()
        for f in ['encals','classes+decay']:
            for m in ['XGBoostMAEPOReg','SkSurvCoxLinear']:
                for db in ['PROACT','ArQ','IDPP']:
                    with open(f'models/gastro_{m}_{f}_{db}.pkl', 'rb') as fid:
                        clf_gastro = pickle.load(fid)

                    print('baseline missingness',"{:.2f}".format(fixed_args[0].isna().sum().sum()/fixed_args[0].size*100),'%')
                    
                    # get random missingness
                    notnans = fixed_args[0].notna()
                    for remove_pct in [.1,.2,.3,.5]:
                        for test_fold_id in range(3):
                            # try:
                            if True:
                                Xnow = fixed_args[0].copy()
                                samples_to_remove = notnans.sum()*remove_pct//1
                                for col in notnans.keys():
                                    idxs = np.where(notnans[col])[0]
                                    np.random.shuffle(idxs)
                                    idxnow = idxs[:int(samples_to_remove[col])]
                                    Xnow.loc[idxnow,col] = np.nan
                                print('baseline missingness',"{:.2f}".format(Xnow.isna().sum().sum()/fixed_args[0].size*100),'%')
                                
                                # get baseline model/data
                                _,Xg,yg,strat_var = select_model_and_features(f,m,Xnow,fixed_args[1],fixed_args[3],fixed_args[4],logger,export_feature_names=False)
                                
                                # get the train test split
                                idx_test = (strat_var==db).reshape(-1,)
                                idx_train = ~idx_test
                                X_test,y_test = Xg[idx_test,:],yg[idx_test]
                                X_train,y_train = Xg[idx_train,:],yg[idx_train]
                                dtrain_valid_combined = clf_gastro.dmat_builder(X_train,y_train)
                                dtest = clf_gastro.dmat_builder(X_test,y_test)

                                # calculate baseline
                                params_df={'learning_rate':0.08,'max_depth': 8,'reg_alpha': 30,'reg_lambda': 5} if m == 'XGBoostMAEPOReg' else {'alpha':0.07}
                                y_pred,y_pred_train = clf_gastro.compute_test_pred(study=None, dtrain_valid_combined=dtrain_valid_combined, dtest=dtest,params_df=params_df)
                            
                                # eval
                                out = calc_all_metrics(pd.DataFrame(),None,clf_gastro,test_fold_id,1300,Xg,yg,y_pred,y_pred_train,np.where(idx_train)[0],np.where(idx_test)[0])
                                out['model'] = m
                                out['features'] = f
                                out['test_database'] = db
                                out['remove_pct'] = remove_pct
                                out['test_fold_id'] = test_fold_id
                                Out = pd.concat([Out,out],axis=0,ignore_index=True)
                                print(m,f,db,remove_pct,test_fold_id)
                                Out.to_csv('data/missingness_test.csv',index=False)
                            # except Exception as e:
                            #     print(e)
        Out.to_csv('data/missingness_test.csv',index=False)
    else:
        Out = pd.read_csv('data/missingness_test.csv')
    
    Out2 = Out.groupby(['model', 'features',  'remove_pct','test_database']).agg(['mean','std']).reset_index()
    Out2.columns = Out2.columns.map('|'.join).str.strip('|')
    Out2['MedianAE_lb'] = Out2['MedianAE_test_uncensored|mean']-Out2['MedianAE_test_uncensored|std']
    Out2['MedianAE_ub'] = Out2['MedianAE_test_uncensored|mean']+Out2['MedianAE_test_uncensored|std']
    base = alt.Chart(Out2).encode(x=alt.X('remove_pct:N', axis=alt.Axis(format='.0%')).title('% points removed'),
                                y=alt.Y('MedianAE_test_uncensored|mean').title('MedianAE'),color=alt.Color('model').scale(domain=list(palette.keys()),range=list(palette.values())))
    dots = base.mark_circle()
    line = base.mark_line().encode(strokeDash='features')
    errorbar = base.mark_errorbar().encode(y=alt.Y('MedianAE_lb').title('MedianAE'),
                                        y2='MedianAE_ub')
    (line+dots+errorbar).facet(column=alt.Column('test_database', title=None)).save('figures/sensitivity_missingness.html')

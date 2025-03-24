import pandas as pd
import os

root = 'Figures/Fig 3/data'
# data panelA
dfA = pd.concat([pd.read_csv(os.path.join(root,p)) for p in os.listdir(root)],axis=0,ignore_index=True)
# data panelB
dfB = pd.concat([pd.read_csv(os.path.join(root,p)) for p in os.listdir(root)],axis=0,ignore_index=True)

# panel A 


# Panel B
cols = ['method','dataset','test_rotation','Cindex_mean',
    'Cindex_lb', 'Cindex_ub', 'MedianAE_mean', 'MedianAE_lb', 'MedianAE_ub',
    'PredIn90_mean', 'PredIn90_lb', 'PredIn90_ub']
models_dfnow = models_df.loc[:,cols]
a = models_dfnow.iloc[:,:6].rename(columns={'Cindex_mean':'value','Cindex_ub':'ub','Cindex_lb':'lb'})
a['variable'] = 'Cindex'
b = models_dfnow.iloc[:,[0,1,2,6,7,8]].rename(columns={'MedianAE_mean':'value','MedianAE_ub':'ub','MedianAE_lb':'lb'})
b['variable'] = 'MedianAE'
c = models_dfnow.iloc[:,[0,1,2,9,10,11]].rename(columns={'PredIn90_mean':'value','PredIn90_ub':'ub','PredIn90_lb':'lb'})
c['variable'] = 'PredIn90'
models_dfnow = pd.concat([a,b,c],axis=0,ignore_index=True)

    # summary variables with confidence bracket
base = alt.Chart(models_dfnow.drop_duplicates(subset=['method','dataset','test_rotation','variable'])).encode(y=alt.Y('dataset').title(None),
                                color = alt.Color('method').scale(domain=list(palette.keys()),range=list(palette.values())),
                                tooltip=['variable','value','ub','lb','method','dataset'])
dots = base.mark_circle(opacity=1).encode(x=alt.X('value',scale=alt.Scale(zero=False)).title(None))
                                
bars = base.mark_errorbar(opacity=1,ticks=True).encode(
                                x=alt.X('lb',scale=alt.Scale(zero=False)).title(None),
                                x2='ub')
(bars + dots).properties(width=200).facet(column=alt.Column('variable').title(None),row=alt.Row('test_rotation').title(None)).resolve_scale(x='independent').save('figures/confidence_bracket2.html')




# supplementary figures SX - comparison of all model metrics

# supplementary figures SX - comparison of all models by individuals

a = 0
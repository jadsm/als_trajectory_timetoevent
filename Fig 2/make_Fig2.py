import pandas as pd
import os
import re
import altair as alt
import numpy as np
from utils import *

# parse the data
colnames = ['mean','lowCI','highCI','time']

def extract_components(filename):
  match = re.match(r'^(\w+)(\d+)(\w+)\.csv$', filename)
  if match:
    return match.groups()
  else:
    return None

def load_csvs(root,file):
    comp = extract_components(file)
    df = pd.read_csv(os.path.join(root,file),header=None,names=colnames)
    df['type'] = comp[0]
    df['class'] = int(comp[1])
    df['variable'] = comp[2]
    return df

# load the data
path_figs = 'data for figures'
df = pd.concat([load_csvs(root,file) for root, folder, files in os.walk(path_figs) for file in files if not file.startswith('weight')],
                       axis=0,ignore_index=True)
# load and condition weight dataframe

dfw = pd.read_csv('data for figures/weight_delay_dataset_cens.csv').rename(columns={'threshold_reached':'class'}).query('threshold==0.05')
dfw['events'] = (dfw['cens'] == 0).astype(int)
results = get_km_values(dfw['tcens'], dfw['events'], dfw['class'])
res0 = results[0]
res0['class'] = 1
res1 = results[1]
res1['class'] = 2
res = pd.concat([res0,res1]).rename(columns={'timeline':'time',
                                               'survival_function':'mean',
                                                   'confidence_interval_lower':'lowCI',
                                                       'confidence_interval_upper':'highCI'})
res['type'] = 'surv'
res['variable'] = 'Weight'
res = res.loc[:,df.keys()]
res['class'] = res['class'].map({1:3,2:1})

dfw2 = pd.read_csv('data for figures/weight_decay_dataset_cens.csv').rename(columns={'days_from_onset':'time','cl2':'class'})
dfw2['type'] = 'traj'
dfw2['variable'] = 'Weight'
dfw2 = dfw2.loc[:,df.keys()].dropna()
dfw2['class'] += 1
dfw2['class'] = dfw2['class'].map({1:3,2:1})
dfw2.loc[:,'mean':'highCI'] = dfw2.loc[:,'mean':'highCI']*100

# dfw = pd.read_csv('data for figures/weight_delay_dataset_final.csv').rename(columns={'threshold_reached':'class'})
# dfw2 = pd.read_csv('data for figures/weight_verticalised.csv').drop(columns=['w0',  'DeltaW',  'Weight']).drop_duplicates().reset_index(drop=True)
# dfw2 = dfw2.groupby(['numid','days_from_onset'])['RelDeltaW'].mean().reset_index().dropna(subset=['RelDeltaW'])
# dfw = dfw2.merge(dfw.loc[:,['numid','class']],on='numid').rename(columns={'days_from_onset':'time'})
# dfwo = dfw.groupby(['time','class'])['RelDeltaW'].describe().reset_index()
# dfwo['lowCI'] = dfwo['mean'] - dfwo['std']/np.sqrt(dfwo['count'])*1.96
# dfwo['highCI'] = dfwo['mean'] + dfwo['std']/np.sqrt(dfwo['count'])*1.96
# dfwo['type'] = 'traj'
# dfwo['variable'] = 'Weight'
# dfwo['class'] = dfwo['class'].map({0:3,1:1})
# dfwo = dfwo.loc[:,df.keys()]
# # concatenate both dataframes
df = pd.concat([df,dfw2,res],axis=0,ignore_index=True).query('time<=1080')

# df.to_csv(os.path.join(path_figs,'traj_surv_merged.csv'),index=False)
df.loc[:,'order'] = df['type'].map({'traj':0,'surv':1})
idx = df['type'] == 'surv'
df.loc[idx,colnames[:3]] = 1 - df.loc[idx,colnames[:3]]
var_map = {'TALS':'Total ALSFRS', 'Bulb':"Bulbar Score", 'q3':"Swallowing function (Q3)", 'Resp':'Respiratory function (FVC)', 'Weight':'Weight'}
df['variable'] = df['variable'].map(var_map)
df.loc[idx,'type'] = 'Prob. Gastrostomy'
df.loc[~idx,'type'] = df.loc[~idx,'variable']

# rename classes
# nclass_dict = df.groupby('variable')['class'].max().to_dict()
# df['nclass'] = df['variable'].map(nclass_dict)

aux = df.query('type == "Prob. Gastrostomy"').groupby(['variable','class'])['mean'].mean().sort_values(ascending=False).reset_index()
aux['newclass'] = aux.groupby('variable')['variable'].cumcount()+1
df = df.merge(aux.loc[:,['class','newclass','variable']],on=['variable',  'class'],how='left')
df['newclass'] = df['newclass'].fillna(df['class'])
df.drop(columns=['class'],inplace=True)
df = df.rename(columns={'newclass':'class'})

class_map = {4:'very slow',3:'slow',2:'medium',1:'fast'}
df['class'] = df['class'].map(class_map)

# {'very slow': #730004, 'slow': #FF9800, 'medium': #00726e, 'rapid':#003E74, 'vrapid':#000074}
# colours = {'very slow':'#730004','slow': '#FF9800', 'medium': '#00726e', 'fast':'#000473'}#, 5:'#000074'
# colours = {'very slow':'#D55E00','slow': '#F0E442', 'medium': '#009E73', 'fast':'#0072B2'}#, 5:'#000074'
colours = {'very slow':'#0072B2','slow': '#94CBEC', 'medium': '#DCCD7D', 'fast':'#C26A77'}#, 5:'#000074'

# plot the data
base = alt.Chart(df).encode(x='time:Q',color = alt.Color('class:N').scale(domain=list(colours.keys()),
                                                                          range=list(colours.values())))
line = base.mark_line().encode(y='mean',
                               tooltip=['mean', 'lowCI', 'highCI', 'time','class'])
errorband = base.mark_errorband().encode(y=alt.Y('lowCI',axis=alt.Axis(tickCount=3)).title(None),
                                         y2 = 'highCI')
chart = (line+errorband).facet(column=alt.Column('type', header=alt.Header(labelFontSize=16)).title(None).sort(alt.SortField(field='order', order='ascending'))).resolve_scale(y='independent')
C = [chart.transform_filter(alt.FieldEqualPredicate(field='variable',equal=var)) for var in df['variable'].unique()]
((C[0]|C[1]|C[4])&(C[2]|C[3])).configure_axis(labelFontSize=16, titleFontSize=16 
).configure_legend(labelFontSize=16,titleFontSize=16,orient='bottom').save('Fig2alt4.html')
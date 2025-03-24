# constants.py
# Author: Juan Delgado-SanMartin
# last reviewed: Jan 2025
# These are constants


dict_number_of_features = {'decayfeatures':10,
                           'demo':9,
                           'classes+decay':10+5,
                           'classes':5,
                           'classes_eq':5,
                           'classes_neq':5,
                           'classes_neqm':5,
                           'demo+classes_neqm':5,
                           'classes_slope':5,
                           'rawfeatures':190,
                           'encals':7,
                           'indiv':1}

palette = {'XGBoostMAEPOReg':'orange','XGBoostCox':'#D55672','SkSurvCoxLinear':'#0075A2','weight':'#8b4513'}


fcn_var = {'ALSFRS_Total': 'exp_decay_fcn', 
           'weight': 'lin_fcn', 
           'q3': 'exp_decay_fcn', 
           'bulbar_subscore': 'exp_decay_fcn', 
           '% predicted': 'exp_decay_fcn'}

funpar = {'ALSFRS_Total': {1: [298.28700706],
                        2: [597.77199464],
                        3: [1221.56585184],
                        4: [2675.51910897]},
        'q3': {1: [424.26751777],
                        2: [1362.81737942],
                        3: [6370.18149953]}, 
        'bulbar_subscore': {1: [444.48237561],
                                2: [1088.03379983],
                                3: [2707.17592451],
                                4: [6307.55946053]},
        '% predicted': {1: [996.24958073],
                        2: [2771.23082],
                        3: [40335.67644986]},
        'weight': {1: [0.00875087],
                     3: [0]}
        }

var_dict = {'Q3_Cl': 'q3', 'Bulb_Cl': 'bulbar_subscore', 'TALS_Cl': 'ALSFRS_Total', 'Resp_Cl': '% predicted'}
classnums = {'q3': 3, 'bulbar_subscore': 4, 'ALSFRS_Total': 4, '% predicted': 3, 'weight': 2}
maxvals = {'% predicted':100, 'ALSFRS_Total':48, 'bulbar_subscore':12, 'q3':4,'weight':0}
var_dict_inv = {v:k for k,v in var_dict.items()}

import os
import re
import sys
import math
import pandas as pd
# import numpy as np
# import pickle
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import multiprocessing
# import threading
import warnings

warnings.filterwarnings('ignore')
print(os.getpid())
rndNum = '672'
uniID = 'P34059'

def biological_feature(df):
    dom=pd.read_csv('../domain.csv')
    rdv=pd.read_csv('../residual_volume.csv')
    dd_residual_vol=[]
    aa_RV_dict=dict(rdv.values)
    
    # Residual Volume
    for i,j in zip(df["Mutant"],df["Wild"]):
        # Mutant Residue
        if len(i) != 1: 
            mut_vol = math.fsum([float(aa_RV_dict.get(x)) for x in [*i]])
        else: 
            mut_vol = aa_RV_dict.get(i)
            
        # Wild Residue
        if len(j) != 1: 
            wild_vol = math.fsum([float(aa_RV_dict.get(x)) for x in [*j]])
        else: 
            wild_vol = aa_RV_dict.get(j)
        dd_residual_vol.append(mut_vol-wild_vol)
    df["dd_changeinresidualvolume"]=dd_residual_vol
    
    # Domain and Modification from Residue to Residue
    dom_dom=dict(dom.drop("Modification",axis=True,inplace=False).values)
    dom_mod=dict(dom.drop("Domain",axis=True,inplace=False).values)
    Domain=[]
    Modification=[]
    for i in list(df["pos"]):
        i=int(i)
        Domain.append(dom_dom.get(i))
        Modification.append(dom_mod.get(i))
    df["Domain"]=Domain
    df["Modification"]=Modification

    # Polarity and Hydrophobicity
    pol=pd.read_csv('../polarity.csv')
    hydrophobicity=pd.read_csv('../hydrophobicity.csv')
    aa_pol_dic=dict(pol.values)
    aa_hydro_dic=dict(hydrophobicity.values)
    dd_pol=[]
    dd_hydrophobicity=[]
    
    # Polarity
    for i,j in zip(df["Mutant"],df["Wild"]):
        # Mutant Residue
        if len(i) != 1:
            mut_pol = math.fsum([float(aa_pol_dic.get(x)) for x in [*i]])
        else: 
            mut_pol = aa_pol_dic.get(i)
            
        # Wild Residue
        if len(j) != 1: 
            wild_pol = math.fsum([float(aa_pol_dic.get(x)) for x in [*j]])
        else: 
            wild_pol = aa_pol_dic.get(j)
        dd_pol.append(mut_pol-wild_pol)
        
    # Hydrophobicity
    for i,j in zip(df["Mutant"],df["Wild"]):
        # Mutant Residue
        if len(i) != 1: 
            mut_hydro = math.fsum([float(aa_hydro_dic.get(x)) for x in [*i]])
        else: 
            mut_hydro = aa_hydro_dic.get(i)
            
        # Wild Residue
        if len(j) != 1: 
            wild_hydro = math.fsum([float(aa_hydro_dic.get(x)) for x in [*j]])
        else: 
            wild_hydro = aa_hydro_dic.get(j)
        dd_hydrophobicity.append(mut_hydro-wild_hydro)

    df["dd_polarity"]=dd_pol
    df["dd_hydrophobicity"]=dd_hydrophobicity
    return df

def create_class(df):
    mutation=df["Mutation"]
    mutation=list(mutation)
    wild=[]
    atlaa=[]
    pos=[]
    for ele in mutation:
        eleLst = re.split('(\d+)',ele)
        wild.append(eleLst[0])
        atlaa.append(eleLst[2])
        pos.append(eleLst[1])
    return([wild,atlaa,pos])

# Preprocessing the dataset
prntDir = os.getcwd().rsplit('/',1)[0]
os.chdir(f"{prntDir}/fileUpload/{rndNum}/outFiles")
tools = sorted(os.listdir())
feature = []
for i in tools:
    feature.append(i.split('.')[0] + '_score')
df = pd.DataFrame()
Mut = pd.read_csv('../GALNS_Mutations_03_11_2023.csv')
df["Mutation"] = Mut["Mutation"]
empty_df = []
for i,j in zip(tools,feature):
    if pd.read_csv(i).empty:
        empty_df = i
    else:
        outF = pd.read_csv(i)
        df[j] =  df['Mutation'].map(outF.set_index('Mutation')['Score'])
        df = df.fillna(0)
mut_info = create_class(df)
df.insert(1,"Wild",mut_info[0])
df.insert(2,"Mutant",mut_info[1])
df.insert(3,"pos",list(map(int, mut_info[2])))
df = biological_feature(df)

# adding Clinical Features to the dataframe
mapping = dict(Mut[['Mutation', 'Phenotype']].values)
df['Phenotype'] = df.Mutation.map(mapping)
df['Phenotype'] = df['Phenotype'].map({"Neutral": 0, "Disease": 1})

# Centering and Scaling using StandardScaler packages to the datasets
X = df.drop(["Mutation","Wild","Mutant","pos","Phenotype"], axis=True).apply(pd.to_numeric, errors = 'coerce')
y = pd.DataFrame(df['Phenotype'], columns=['Phenotype']).apply(pd.to_numeric, errors = 'coerce')
ss = StandardScaler()
X = pd.DataFrame(ss.fit_transform(X),columns=X.columns)

# Hyperparameter Tuning for LogisticsRegression, RandomForest, AdaBoostClassifier, SVM
emptyDict = {"RandomForest": "", "LogisticsRegression": "", "AdaBoostClassifier": "", "SVM": ""}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
def Logistics_Regression_bestparameter():
    estimator = LogisticRegression()
    parameter = {'penalty': ["l1", "l2", "elasticnet"], 'C': [1,2,3,4,5,10,20,30,50,100,300], 'max_iter': [100, 200, 500, 800, 1000], 
                 'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], 
                 'class_weight': ['balanced', 'balanced_subsample'], 'fit_intercept': [True, False], 
                 'random_state': [42, 2021, 1234]}
    classifier_regressor = GridSearchCV(estimator, param_grid=parameter, scoring="accuracy", cv=cv)
    classifier_regressor.fit(X, y)
    emptyDict['LogisticsRegression'] = classifier_regressor.best_params_
    print(classifier_regressor.best_params_)
    
def RandomForestClassifier_bestparameter():
    estimator = RandomForestClassifier()
    parameter = {'n_estimators': [50,100,200,400,800], 'criterion': ['gini', 'entropy'], 'max_depth': [None,2,5,7,9,10],
                 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4], 'class_weight': ['balanced', 'balanced_subsample'],
                 'random_state': [42, 70, 150, 200], 'max_features': ['auto', 'sqrt', 'log2', None], 'n_jobs': [4]}
    classifier_regressor = GridSearchCV(estimator, param_grid=parameter, scoring="accuracy", cv=cv)
    classifier_regressor.fit(X, y)
    emptyDict['RandomForest'] = classifier_regressor.best_params_
    print(classifier_regressor.best_params_)
    
def AdaBoostClassifier_bestparameter():
    estimator = AdaBoostClassifier()
    parameter = {'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
                 'n_estimators': [50, 100, 200], 'random_state': [42, 70, 150, 200], 'learning_rate': [0.1, 0.5, 1.0],
                 'algorithm': ['SAMME', 'SAMME.R']}
    classifier_regressor = GridSearchCV(estimator, param_grid=parameter, scoring="accuracy", cv=cv)
    classifier_regressor.fit(X, y)
    emptyDict['AdaBoostClassifier'] = classifier_regressor.best_params_
    print(classifier_regressor.best_params_)

def SVMClassifier_bestparameter():
    estimator = svm.SVC()
    parameter = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf', 'poly'], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto'], 
                 'shrinking': [True, False], 'probability': [True, False], 'max_iter': [100, 500, 1000], 
                 'decision_function_shape': ['ovr', 'ovo'], 'cache_size': [200, 500], 'tol': [1e-3, 1e-4]}
    classifier_regressor = GridSearchCV(estimator, param_grid=parameter, scoring="accuracy", cv=cv)
    classifier_regressor.fit(X, y)
    emptyDict['SVM'] = classifier_regressor.best_params_
    print(classifier_regressor.best_params_)

print("Processing has been started")

# Start the prcessing
pool_1 = multiprocessing.Pool(processes=2)
result_1 = pool_1.apply_async(Logistics_Regression_bestparameter)

pool_2 = multiprocessing.Pool(processes=2)
result_2 = pool_2.apply_async(RandomForestClassifier_bestparameter)

pool_3 = multiprocessing.Pool(processes=2)
result_3 = pool_3.apply_async(AdaBoostClassifier_bestparameter)

pool_4 = multiprocessing.Pool(processes=2)
result_4 = pool_4.apply_async(SVMClassifier_bestparameter)

Close the pool
pool_1.close()
pool_1.join()

pool_2.close()
pool_2.join()

pool_3.close()
pool_3.join()

pool_4.close()
pool_4.join()

filehandler = open("../HyperParameter_Classifier", 'wt')
filehandler.write(str(emptyDict))
filehandler.close()

print("Successfully Done")
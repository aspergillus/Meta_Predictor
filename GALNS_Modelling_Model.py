import os
import re
import sys
import math
import warnings
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import recall_score, precision_score, f1_score, matthews_corrcoef, accuracy_score

# Ignoring the warnings
warnings.filterwarnings('ignore')

# Pre-processing the DataSet
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

X = df.drop(["Mutation","Wild","Mutant","pos","Phenotype"], axis=True)
y = pd.DataFrame(df['Phenotype'], columns=['Phenotype'])

######## Modelling the Model ######### 
# Base estimators for stacking
base_estimators = [
    ("rf", RandomForestClassifier(class_weight='balanced_subsample', max_depth=9, min_samples_leaf=2, min_samples_split=5, n_estimators=100, random_state=200)),
    ("svm", SVC(C=0.1, kernel='linear', probability=True, random_state=42)),
    ("adaboost", AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2), n_estimators=100, random_state=42))
]

# Final stacking classifier
Stacker = StackingClassifier(estimators=base_estimators, final_estimator=LogisticRegression(C=1, class_weight='balanced', penalty='l1', random_state=42, solver='liblinear'))

# Classifiers list
Classifiers = [
    LogisticRegression(C=1, class_weight='balanced', fit_intercept=True, max_iter=100, penalty='l1', random_state=42, solver='saga'),
    RandomForestClassifier(class_weight='balanced_subsample', criterion='entropy', max_depth=7, max_features='auto', min_samples_leaf=4, 
                                  min_samples_split=2, n_estimators=100, n_jobs=4, random_state=200),
    AdaBoostClassifier(algorithm='SAMME', base_estimator=DecisionTreeClassifier(max_depth=2), learning_rate=1.0, n_estimators=50, random_state=42),
    SVC(C=1, cache_size=200, decision_function_shape='ovr', degree=2, gamma='scale', kernel='rbf', max_iter=100, probability=True, shrinking=True, tol=0.001),
    Stacker
]

# Initialize empty lists to store accuracies and models
c = 0
test_data_dict = {}
accuracies, models, standScalerList = ([] for i in range(3))
Classi_Name = ["LogisticRegression", "RandomForest", "AdaBoost", "SVM", "Stacking"]
Short_Model_Name = ["Logistic", "RandomForest", "AdaBoost", "SVM", "Stack"]

LR_Model, RF_Model, AdB_Model, SVM_Classi_Model, Stack_Classi_Model = ([] for i in range(5))
LR_df, RF_df, AdB_df, SVM_df, Stack_df = (pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "F1 score", "MCC"]) for i in range(5))
Model_df = [LR_df, RF_df, AdB_df, SVM_df, Stack_df]

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    test_data_dict[c] = X_test.index

    ss = StandardScaler()
    X_train = pd.DataFrame(ss.fit_transform(X_train),columns = X_train.columns)
    X_test = pd.DataFrame(ss.transform(X_test),columns = X_test.columns)
    
    standScalerList.append(ss)
    classi_c = 0
    for model in Classifiers:
        def check_model(model, xTrain, yTrain, xTest):
            # Train the model on the training data
            model.fit(xTrain, yTrain)

            # Evaluate the model's performance on the test data
            predictions = model.predict(xTest)
            accur = accuracy_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            precision = precision_score(y_test, predictions)
            f1_Score = f1_score(y_test, predictions)
            mcc = matthews_corrcoef(y_test, predictions)
            return model, accur, recall, precision, f1_Score, mcc
        
        # Store the accuracy and model
        if classi_c is 0:
            model, accur, recall, preci, f1, mcc  = check_model(model, X_train, y_train, X_test)
            LR_Model.append(model)
            LR_df.loc[len(LR_df)] = accur, recall, preci, f1, mcc
        elif classi_c is 1:
            model, accur, recall, preci, f1, mcc  = check_model(model, X_train, y_train, X_test)
            RF_Model.append(model)
            RF_df.loc[len(LR_df)] = accur, recall, preci, f1, mcc
        elif classi_c is 2:
            model, accur, recall, preci, f1, mcc  = check_model(model, X_train, y_train, X_test)
            AdB_Model.append(model)
            AdB_df.loc[len(LR_df)] = accur, recall, preci, f1, mcc
        elif classi_c is 3:
            model, accur, recall, preci, f1, mcc  = check_model(model, X_train, y_train, X_test)
            SVM_Classi_Model.append(model)
            SVM_df.loc[len(LR_df)] = accur, recall, preci, f1, mcc
        elif classi_c is 4:
            model, accur, recall, preci, f1, mcc  = check_model(model, X_train, y_train, X_test)
            Stack_Classi_Model.append(model)
            Stack_df.loc[len(LR_df)] = accur, recall, preci, f1, mcc
        classi_c = classi_c + 1
    c = c + 1
    
Model_Accur_Lst = [LR_df["Accuracy"], RF_df["Accuracy"], AdB_df["Accuracy"], SVM_df["Accuracy"], Stack_df["Accuracy"]]
Model_Lst = [LR_Model, RF_Model, AdB_Model, SVM_Classi_Model, Stack_Classi_Model]
best_model_number_lst = []

# Identify the best model among all of 
for accur, model, model_name in zip(Model_Accur_Lst, Model_Lst, Classi_Name):
    highest_accuracy = max(accur)
    Accur_Dict = dict(zip((i for i in range(5)), accur))
    best_model_number = [model for model, accuracy in Accur_Dict.items() if accuracy == highest_accuracy][0]
    best_model_number_lst.append(best_model_number)
    
    # Export the best model
    with open(f'../{model_name}_Model.pkl', 'wb') as f:
        pickle.dump(model[best_model_number], f)
pickle.dump(standScalerList[mode(best_model_number_lst)], open("../Scaler_MPS_IV_A.pkl", 'wb'))
test_data_validation = df.iloc[test_data_dict[mode(best_model_number_lst)]]
test_data_validation.to_csv("../Test_Dataset.csv", index=False)

# Identify the Best Model_Score among all Models
Long_Model_Name = ["Logistic Regession", "Random Forest", "AdaBoost", "Support Vector Machine", "Stacking"]
Model_Accuracy = pd.DataFrame(columns=["Model", "Accuracy", "Recall", "Precision", "F1 score", "MCC"])
for model_acc, model_name in zip(Model_df, Long_Model_Name):
    model_score = model_acc.iloc[model_acc["Accuracy"].idxmax()]
    Model_Accuracy.loc[len(Model_Accuracy)] = model_name, model_score[0], model_score[1], model_score[2], model_score[3], model_score[4]
Model_Accuracy.to_csv("../Meta_Predictor_Metrics.csv", index=False)
import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# rndNum = 525
rndNum = sys.argv[1]
prntDir = os.getcwd()
os.chdir(f"{prntDir}/fileUpload/{rndNum}")

df=pd.read_csv("train_set.csv")
X = df.drop(["Mutation","Wild","Mutant","pos","clinical_phenotype"],axis=True)
y = pd.DataFrame(df['clinical_phenotype'], columns=['clinical_phenotype'])
ConvX = X.apply(pd.to_numeric, errors = 'coerce')
ConvY = y.apply(pd.to_numeric, errors = 'coerce')

# Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(ConvX, ConvY, test_size=0.2, random_state=42)

# standardizing the data
ss=StandardScaler()
ss_train=pd.DataFrame(ss.fit_transform(X_train),columns=X_train.columns)
ss_test=pd.DataFrame(ss.transform(X_test),columns=X_test.columns)

# Model
def ML(model,xtrain,ytrain,xtest):
    test_mut=pd.DataFrame()
    model.fit(xtrain,ytrain)
    y_predicted=model.predict(xtest)
    dumm=np.where(y_predicted==0,"Neutral","Severe")
    test_mut["predicted"]=dumm
    prob=model.predict_proba(xtest)
    LR_prob=[]
    for i,j in zip(y_predicted,prob):
        LR_prob.append(j[i])
    test_mut["probability"]=LR_prob
    return test_mut

def getMutation(indexNum):
    res = df['Mutation'][indexNum]
    return res

#### LR #########
lr=LogisticRegression(penalty='l2',dual=False,tol=1e-8,
                      C=0.05,fit_intercept=True,class_weight={0:0.957,1:2.6},random_state=14,solver='liblinear')
lr=ML(lr,ConvX,ConvY,ss_test)
result = pd.DataFrame(map(getMutation, sorted(X_test.index)), columns=['Mutation'])
final_df=pd.concat([result,lr],axis=1)
final_df.to_csv("ModelPrediction.csv",index=False)

# Below this code is used for the exporting the model
# model=LogisticRegression(penalty='l2',dual=False,tol=1e-8,
#                       C=0.05,fit_intercept=True,class_weight={0:0.957,1:2.6},random_state=14,solver='liblinear')
# test_mut=pd.DataFrame()
# model.fit(ConvX,ConvY)
# import pickle
# pickle.dump(model, open('MPS_III_A.pkl', 'wb'))
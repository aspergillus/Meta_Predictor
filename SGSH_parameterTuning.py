import os
import re
import math
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import GridSearchCV, KFold
import multiprocessing
import warnings

warnings.filterwarnings('ignore')
print(os.getpid())

rndNum = '931'
uniID = 'P51688'

def biological_feature(df):
    dom = pd.read_csv('../domain.csv')
    rdv = pd.read_csv('../residual_volume.csv')
    aa_RV_dict = dict(rdv.values)

    dd_residual_vol = [
        (math.fsum([float(aa_RV_dict.get(x)) for x in [*i]]) if len(i) != 1 else aa_RV_dict.get(i)) -
        (math.fsum([float(aa_RV_dict.get(x)) for x in [*j]]) if len(j) != 1 else aa_RV_dict.get(j))
        for i, j in zip(df["Mutant"], df["Wild"])
    ]
    df["dd_changeinresidualvolume"] = dd_residual_vol

    dom_dom = dict(dom.drop("Modification", axis=True).values)
    dom_mod = dict(dom.drop("Domain", axis=True).values)
    df["Domain"] = df["pos"].map(dom_dom)
    df["Modification"] = df["pos"].map(dom_mod)

    pol = pd.read_csv('../polarity.csv')
    hydrophobicity = pd.read_csv('../hydrophobicity.csv')
    aa_pol_dic = dict(pol.values)
    aa_hydro_dic = dict(hydrophobicity.values)

    df["dd_polarity"] = [
        (math.fsum([float(aa_pol_dic.get(x)) for x in [*i]]) if len(i) != 1 else aa_pol_dic.get(i)) -
        (math.fsum([float(aa_pol_dic.get(x)) for x in [*j]]) if len(j) != 1 else aa_pol_dic.get(j))
        for i, j in zip(df["Mutant"], df["Wild"])
    ]
    df["dd_hydrophobicity"] = [
        (math.fsum([float(aa_hydro_dic.get(x)) for x in [*i]]) if len(i) != 1 else aa_hydro_dic.get(i)) -
        (math.fsum([float(aa_hydro_dic.get(x)) for x in [*j]]) if len(j) != 1 else aa_hydro_dic.get(j))
        for i, j in zip(df["Mutant"], df["Wild"])
    ]

    return df

def create_class(df):
    mutation = df["Mutation"]
    wild, atlaa, pos = zip(*[(re.split('(\d+)', ele)[0], re.split('(\d+)', ele)[2], re.split('(\d+)', ele)[1]) for ele in mutation])
    return wild, atlaa, pos

prntDir = os.getcwd().rsplit('/', 1)[0]
os.chdir(f"{prntDir}/fileUpload/{rndNum}/outFiles")
tools = sorted(os.listdir())
features = [i.split('.')[0] + '_score' for i in tools]
df = pd.DataFrame()

Mut = pd.read_csv('../SGSH_Mutations_12_06_2024.csv').drop_duplicates('Mutation')
df["Mutation"] = Mut["Mutation"]

for i, feature in zip(tools, features):
    if not pd.read_csv(i).empty:
        df[feature] = df['Mutation'].map(pd.read_csv(i).set_index('Mutation')['Score'])
        df.fillna(0, inplace=True)

wild, mutant, pos = create_class(df)
df.insert(1, "Wild", wild)
df.insert(2, "Mutant", mutant)
df.insert(3, "pos", list(map(int, pos)))
df = biological_feature(df)

mapping = dict(Mut[['Mutation', 'Phenotype']].values)
df['Phenotype'] = df['Mutation'].map(mapping).map({"Neutral": 0, "Disease": 1})

X = df.drop(["Mutation", "Wild", "Mutant", "pos", "Phenotype"], axis=True).apply(pd.to_numeric, errors='coerce')
y = df['Phenotype'].apply(pd.to_numeric, errors='coerce')
X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

cv = KFold(n_splits=10, shuffle=True, random_state=42)

param_grid = {
    'LogisticRegression': {
        'penalty': ["l1", "l2", "elasticnet"],
        'C': [1, 2, 3, 4, 5, 10, 20, 30, 50, 100, 300],
        'max_iter': [100, 200, 500, 800, 1000],
        'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        'class_weight': ['balanced', 'balanced_subsample'],
        'fit_intercept': [True, False],
        'random_state': [42, 2021, 1234]
    },
    'RandomForestClassifier': {
        'n_estimators': [50, 100, 200, 400, 800],
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 2, 5, 7, 9, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
        'random_state': [42, 70, 150, 200],
        'max_features': ['auto', 'sqrt', 'log2', None],
        'n_jobs': [4]
    },
    'AdaBoostClassifier': {
        'base_estimator': [DecisionTreeClassifier(max_depth=1), DecisionTreeClassifier(max_depth=2)],
        'n_estimators': [50, 100, 200],
        'random_state': [42, 70, 150, 200],
        'learning_rate': [0.1, 0.5, 1.0],
        'algorithm': ['SAMME', 'SAMME.R']
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
        'shrinking': [True, False],
        'probability': [True, False],
        'max_iter': [100, 500, 1000],
        'decision_function_shape': ['ovr', 'ovo'],
        'cache_size': [200, 500],
        'tol': [1e-3, 1e-4]
    }
}

emptyDict = {key: "" for key in param_grid.keys()}

def tune_model(model_name, estimator, param_grid):
    classifier_regressor = GridSearchCV(estimator, param_grid=param_grid, scoring="accuracy", cv=cv)
    classifier_regressor.fit(X, y)
    emptyDict[model_name] = classifier_regressor.best_params_
    print(classifier_regressor.best_params_)

model_mapping = {
    'LogisticRegression': LogisticRegression(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'SVM': svm.SVC()
}

print("Processing has started")

with multiprocessing.Pool(processes=2) as pool:
    results = [
        pool.apply_async(tune_model, (name, model_mapping[name], param_grid[name]))
        for name in model_mapping.keys()
    ]
    for result in results:
        result.get()

with open("../SGSH_HyperParameter_Classifier", 'wt') as filehandler:
    filehandler.write(str(emptyDict))

print("Successfully Done")
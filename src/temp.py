import pickle
from sklearn.model_selection import cross_val_score,StratifiedKFold,GridSearchCV
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.svm import LinearSVC,SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import os



emb_path = os.path.join(os.path.join("..",'data'),"data.pickle")
with open(emb_path,'rb') as f:
    data = pickle.load(f)

X = data['embeddings']
y = data['labels']

print('Creating data for ML model')
np.random.seed(42)
scalar = StandardScaler()
X = np.array(X)
y = np.array(y)
indexes = np.random.permutation(np.arange(X.shape[0]))
le = LabelEncoder()
y = le.fit_transform(y)
X = X[indexes]
y = y[indexes]
#model = LinearSVC()
model = SVC()
pipeline = Pipeline([('transformer', scalar), ('estimator',model)])
cv = StratifiedKFold(n_splits=5)
param_grid = {'estimator__C':[0.0001,0.001,0.01,0.1,1,10],'estimator__kernel':['linear','rbf'],'estimator__gamma':['auto','scale']}
grid_svc = GridSearchCV(pipeline,param_grid = param_grid,scoring = 'accuracy',verbose = True,cv = cv,n_jobs = -1)
grid_svc.fit(X,y)
print("The Best score for svc model is {} with params {}".format(grid_svc.best_score_,grid_svc.best_params_))
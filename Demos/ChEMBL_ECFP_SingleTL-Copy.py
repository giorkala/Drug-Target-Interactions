#!/usr/bin/env python
# coding: utf-8

# ## Single Task Learning prediction for kinases with data from ChEMBL
# This notebook evaluates and compares the performance of three regressors, Random Forests, Neural Networks, and Lasso Regression, all implemented by Scikit-learn for predicting bioactivity values. We are using pIC50 values on a dataset with 110 targets. As we are not interested for optimisation at this point, we run just a quick parameter selection through a 4-fold cross-validation.

# ## 0. Prerequisites

# In[10]:


import numpy as np
import pandas as pd
import keras
import os, pickle

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE

from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor

from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.wrappers.scikit_learn import KerasRegressor # this is for making a model like every other in scikit
from keras.models import model_from_json

import  matplotlib.pyplot as plt

TRAINEDMODELS = 'TrainedModels'
random_seed = 2019
np.random.seed(random_seed)
nfolds=4
njobs =3
# RESCALE_FACTOR = 10.0 # 0-10000 -> 0-1000 # no need for that, we use -logIC50 instead.


# ### Load data file from disk 

# In[2]:


Interactions_train = []    
with open("Interactions_Trainset.tab",'r') as f:
    for line in f:
        tokens = line.split()
        # 'Target-ID', 'Compound-ID', 'pIC50'  
        Interactions_train.append( [tokens[0], tokens[1], float(tokens[2]) ])

Interactions_valid = []        
with open("Interactions_Validset.tab",'r') as f:
    for line in f:
        tokens = line.split()
        # 'Target-ID', 'Compound-ID', 'pIC50'  
        Interactions_valid.append( [tokens[0], tokens[1], float(tokens[2]) ])

Interactions = [x for x in Interactions_train]
Interactions.extend(Interactions_valid)
# we use a dataframe to quickly sort targets wrt #compounds:
DF = pd.DataFrame( Interactions, columns =['Target-ID', 'Compound-ID','Std-value']) 
temp = DF.groupby(['Target-ID']).agg('count').sort_values(by='Compound-ID') # count the number of molecules
Targets = list(temp.index)
Compounds = np.unique(DF['Compound-ID'])

nT=len(Targets); nC=len(Compounds)

print("There are {0} targets and {1} compounds currently loaded with {2} interactions.".format(nT,nC,len(Interactions)))
print("A DTI matrix would be {0:.4}% dense!".format(100.0*len(Interactions)/nT/nC ))

# first we need to prepare each fp as a feature vector
Fingerprints={} # this contains one list per fingerprint - not efficient...
with open('Compound_Fingerprints.tab', 'r') as f:
    header = f.readline()
    for line in f:
        # each line is Comp-ID, SMILES, FP
        tokens = line.split()
        # we keep only those compounds which have FPs
        if tokens[2] != 'NOFP':
            fp = [int(c) for c in tokens[2] ]
            Fingerprints[ tokens[0] ] = fp
print("%d fingerprints were loaded!" % len(Fingerprints))


# ## 4. Personalised NN with Keras

# In[23]:


myNN_all = dict()
Scores_myNN_train=[]
param_grid={'lamda':[0.2, 0.1, 0.01, 0.001]}
count=0

def mymodel(lamda, init=-4.5):
    model = Sequential()

    model.add(Dense(units=100, activation='relu', kernel_regularizer=regularizers.l2(lamda), input_dim=2048))
    model.add(Dense(units=20,  activation='relu', kernel_regularizer=regularizers.l2(lamda), input_dim=100 ))
    myinit = keras.initializers.Constant(value=init)
    model.add(Dense(1, kernel_initializer=myinit, activity_regularizer=regularizers.l1(0.001)))
    
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.adam(lr=0.001))
    return model

for target in Targets:
    # define the train set
    X_train=[]; Y_train=[]
    for point in Interactions_train:
        if point[0]==target:
            X_train.append( Fingerprints[point[1]] )
            Y_train.append( float(point[2]) )
        
    X_train = np.array( X_train )
    if os.path.isfile(TRAINEDMODELS+'/Keras/Keras_'+target+'_'+'pIC50model.sav'):
        # model is already trained for current target
        with open( TRAINEDMODELS+'/Keras/Keras_'+target+'_'+'pIC50model.sav', 'r') as jsonf:
            json_model = jsonf.read()
        myNN = model_from_json(json_model)
        myNN.load_weights( TRAINEDMODELS+'/Keras/Keras_'+target+'_'+'weights.h5'  )
        myNN.compile(loss='mean_squared_error', optimizer=keras.optimizers.adam(lr=0.001))
        myNN.fit(X_train,Y_train, epochs=250, batch_size=20, verbose=0)
    else:
        myNN = KerasRegressor(build_fn=mymodel, init=-4.5, epochs=250, batch_size=20, verbose=0)
        # fit model to data:
        cvr = GridSearchCV(myNN, param_grid=param_grid, cv=nfolds, n_jobs=njobs, iid=True)
        cvr.fit(X_train, Y_train)
        myNN = KerasRegressor(build_fn=mymodel, init=-4.5, lamda=cvr.best_params_['lamda'], epochs=250, batch_size=20, verbose=0)
        myNN.fit(X_train,Y_train)
        # save:
        model_json=myNN.to_json()
        with open( TRAINEDMODELS+'/Keras/Keras_'+target+'_'+'pIC50model.json', 'w') as jsonf:
            jsonf.write(model_json)
        myNN.save_weights( TRAINEDMODELS+'/Keras/Keras_'+target+'_'+'weights.h5' )
    Y_NN = myNN.predict(X_train)
    # get scores and details:
    Scores_myNN_train.append( r2_score(Y_train, Y_NN) )
    Target_info[target]['my_train_r2'] = Scores_myNN_train[-1] # add info
#   print("R2 score for {0} with {1} items = {2:.3f}".format(target, len(Y_train), Scores_myNN_train[-1]))
    myNN_all[target] = myNN # save model for validation

    if count%10==0:
        print("More than %d targets are processed" % count)
        print("Mean score so far: %f" % np.mean(Scores_myNN_train))
    count+=1
    
print("Mean score for myNN during training = %f" % np.mean(Scores_myNN_train) )


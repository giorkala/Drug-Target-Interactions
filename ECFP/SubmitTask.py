#-----------------------------------------------------------------#
# 1. Train/Validationset are files 
# 2. Target should be the ID of current protein
# 3. We assume that FP are stored at 'Compound_Fingerprints.tab'
# 4. We also get the list of compounds from the that file
#-----------------------------------------------------------------#
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import pickle,sys
import numpy as np

nfolds=3; njobs =5
random_seed = 2019; np.random.seed(random_seed)
Target = sys.argv[3]

def Evaluate( TARGET, MODEL, validationset ):
    True_temp = []; Pred_temp = []
    with open( validationset, 'r') as file:
        # no header on this file
        for line in file:
            tokens = line.split()
            if tokens[0]==TARGET:
                True_temp.append( float(tokens[2]) )
                x_test = np.array( Fingerprints[tokens[1]] ).reshape(1,-1)
                Pred_temp.append( MODEL.predict( x_test ) )
    r2 = r2_score(True_temp,Pred_temp)
    print("R2-score after {0} points = {1:.4f} ".format(len(True_temp), r2 ) )
    return r2

######################################################
# first we need to prepare each fp as a feature vector
Fingerprints={} # this contains one list per fingerprint 
Compounds = []
with open('../Compound_Fingerprints.tab', 'r') as f:
    header = f.readline()
    for line in f:
        # each line is Comp-ID, SMILES, FP
        tokens = line.split()
        # we keep only those compounds which have FPs
        if tokens[2] != 'NOFP':
            fp = [int(c) for c in tokens[2] ]
            Fingerprints[ tokens[0] ] = fp
            Compounds.append( tokens[0] )
print("%d fingerprints were loaded." % len(Fingerprints))

##################
# Load train set #
#Target_info = {}
X_train=[]; Y_train=[]
with open( sys.argv[1], 'r') as file:
    # no header on this file
    for line in file:
        tokens = line.split()
        # 'Target-ID', 'Compound-ID', 'pIC50'
        if tokens[0]==Target:
            X_train.append( Fingerprints[tokens[1]] )
            Y_train.append( float(tokens[2]) )
print("Number of loaded interactions = %d" % len(Y_train) )
#Target_info['train_size']=len(Y_train) # add info       

###########################
# param-selection with cv #
print("Selecting parameters with CV")
param_grid={'n_estimators':[10,25,50,100,150], 'max_depth':[3,4,5,7,10,15,20], 'max_features':['sqrt','auto']}
cvr = GridSearchCV(RandomForestRegressor(random_state=2019), param_grid, cv=nfolds, n_jobs=njobs, iid=True)
cvr.fit(X_train, Y_train)
# select best parametrisation and train to the complete train-set
RFR = RandomForestRegressor( n_estimators= cvr.best_params_['n_estimators'],max_features=cvr.best_params_['max_features'], max_depth=cvr.best_params_['max_depth'], random_state=2019)

RFR.fit(X_train,Y_train)
#Target_info["model"] = RFR
#Target_info['RF_train_r2'] = RFR.score( X_train,  Y_train) # add info
print("Training score after {0} points = {1:.4f} ".format(len(Y_train), RFR.score( X_train,  Y_train) ) )
filename = 'TrainedModels/RF_'+Target+'_'+'pIC50new.sav'
pickle.dump(RFR, open(filename, 'wb'))

########################################
# now evaluate with the validation set #
True_temp = []; Pred_temp = []
with open( sys.argv[2], 'r') as file:
    # no header on this file
    with open(Target+".predictions.txt", 'w') as fnew:
        for line in file:
            tokens = line.split()
            if tokens[0]==Target:
                True_temp.append( float(tokens[2]) )
                x_test = np.array( Fingerprints[tokens[1]] ).reshape(1,-1)
                Pred_temp.append( RFR.predict( x_test ) )
                # save some results #
                fnew.write( "{0}\t{1}\t{2}\t{3}\n".format( Target, tokens[1], tokens[2], Pred_temp[-1]) )
print("R2-score after {0} points = {1:.4f} ".format(len(True_temp), r2_score(True_temp,Pred_temp) ) )
#####################
